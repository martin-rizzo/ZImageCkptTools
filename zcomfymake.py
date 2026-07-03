"""
File    : convmodel.py
Purpose : Script to convert the original Z-Image checkpoint to a format compatible with ComfyUI
Author  : Martin Rizzo | <martinrizzo@gmail.com>
Date    : Jun 26, 2026
Repo    : https://github.com/martin-rizzo/ComfyUI-ZImagePowerNodes
License : MIT
- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                          ComfyUI-ZImagePowerNodes
     ComfyUI nodes designed to power the "Z-Image/Z-Image Turbo" models.
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _
"""
import io
import os
import sys
import json
import tempfile
import argparse
from pathlib import Path
from typing import IO
if __name__ == '__main__' and ("-h" not in sys.argv and "--help" not in sys.argv):
    # modules that are not available in the standard library are imported here
    import numpy as np
    import ml_dtypes
    from typing      import Callable, Any
    from safetensors import safe_open

# Safetensors Header Structure
# each tensor name maps to a dictionary containing its metadata
type SafetensorsHeader = dict[str, dict[str, Any]]


# get the directory where the script is located
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# ANSI escape codes for colored terminal output
RED      = '\033[91m'
DKRED    = '\033[31m'
YELLOW   = '\033[93m'
DKYELLOW = '\033[33m'
GREEN    = '\033[92m'
CYAN     = '\033[96m'
DKGRAY   = '\033[90m'
RESET    = '\033[0m'

#============================= ERROR MESSAGES ==============================#

def disable_colors():
    global RED, DKRED, YELLOW, DKYELLOW, GREEN, CYAN, DKGRAY, RESET
    RED, DKRED, YELLOW, DKYELLOW, GREEN, CYAN, DKGRAY, RESET = "", "", "", "", "", "", "", ""


def message(msg: str, *extra_messages: str, padding: int = 0, file=sys.stderr) -> None:
    """Displays a regular message with custom padding."""
    pad_spaces = " " * padding
    if msg:
        print(f"{pad_spaces}{GREEN}>{RESET} {msg}", end="", file=file)
        for extra in extra_messages:
            print(f" {extra}", end="", file=file)
    print(file=file)


def warning(msg: str, *info_messages: str, padding: int = 0, file=sys.stderr) -> None:
    """Displays a warning message and its info messages with custom padding."""
    pad_spaces = " " * padding
    print(f"{pad_spaces}{CYAN}[{YELLOW}WARNING{CYAN}]{YELLOW} {msg}{RESET}", file=file)
    for info_message in info_messages:
        print(f"{pad_spaces} {CYAN}\xF0\x9F\x9B\x88 {info_message}{RESET}", file=file)


def error(msg: str, *info_messages: str, padding: int = 0, file=sys.stderr) -> None:
    """Displays an error message and its info messages with custom padding."""
    pad_spaces = " " * padding
    print(f"{pad_spaces}{CYAN}[{RED}ERROR{CYAN}]{RED} {msg}{RESET}", file=file)
    for info_message in info_messages:
        print(f"{pad_spaces} {CYAN}\xF0\x9F\x9B\x88 {info_message}{RESET}", file=file)


#============================== PROGRESS BAR ===============================#

class ProgressBar:
    """
    A simple and efficient progress bar.
    Args:
        minimum : The starting value of the progress range.
        maximum : The ending value of the progress range.
        message : A prefix string to display before the bar.
        length  : The visual length of the progress bar in characters.
        file    : The output stream (e.g., sys.stdout or sys.stderr).
    """
    def __init__(self,
                 minimum: float = 0.0,
                 maximum: float = 1.0,
                 message: str   = "Progress",
                 length : int   = 30,
                 file           = sys.stdout) -> None:
        self.minimum = minimum
        self.range   = maximum - minimum
        self.message = message
        self.length  = length
        self.file    = file

    def update(self, progress: float) -> None:
        """
        Updates and redraws the progress bar to the file stream.

        Args:
            progress : Current value to calculate the completion percentage.
        """
        # Normalize progress between 0.0 and 1.0
        fraction = max(0.0, min(1.0, (progress - self.minimum) / self.range))
        filled_length = int(self.length * fraction)

        # Create visual bar
        bar = '█' * filled_length + '-' * (self.length - filled_length)
        percent = f"{fraction * 100:3.1f}%"

        # Use carriage return '\r' to overwrite the line in the console
        self.file.write(f"\r{self.message} |{bar}| {percent}")
        self.file.flush()

        # Print a newline when complete
        if fraction == 1.0:
            self.file.write("\n")


#============================== TENSOR MAPPER ==============================#

class TensorMapper:
    def __init__(self):
            self.replace_keys = {
                "all_final_layer.2-1."      : "final_layer.",
                "all_x_embedder.2-1."       : "x_embedder.",
                ".attention.to_out.0.bias"  : ".attention.out.bias",
                ".attention.norm_k.weight"  : ".attention.k_norm.weight",
                ".attention.norm_q.weight"  : ".attention.q_norm.weight",
                ".attention.to_out.0.weight": ".attention.out.weight"
            }
            self.qkv_buffers = {}

    def __call__(self, tensor_name: str, tensor: np.ndarray):

        # omit tensors that cause conflicts or are unnecessary
        if tensor_name.endswith(".attention.to_out.0.bias"):
            return None, None

        # extract the layer prefix (e.g., "model.layers.0")
        layer_prefix = tensor_name.rsplit(".attention.", 1)[0] if ".attention." in tensor_name else ""

        # logic for accumulating QKV tensors
        if tensor_name.endswith(".attention.to_k.weight"):
            if layer_prefix not in self.qkv_buffers: self.qkv_buffers[layer_prefix] = {}
            self.qkv_buffers[layer_prefix]['k'] = tensor
            return None, None

        if tensor_name.endswith(".attention.to_q.weight"):
            if layer_prefix not in self.qkv_buffers: self.qkv_buffers[layer_prefix] = {}
            self.qkv_buffers[layer_prefix]['q'] = tensor
            return None, None

        if tensor_name.endswith(".attention.to_v.weight"):

            # once V arrives, the previous Q and K components should exist
            qkv = self.qkv_buffers.get(layer_prefix, {})
            if 'q' in qkv and 'k' in qkv:
                # concatenate Q, K, and V into a single tensor QKV and clear the buffer for this layer
                tensor      = np.concatenate([qkv['q'], qkv['k'], tensor], axis=0)
                tensor_name = tensor_name.replace(".attention.to_v.weight", ".attention.qkv.weight")
                del self.qkv_buffers[layer_prefix]

            # if V arrives before Q and K, raise an error and exit
            # (the support for qkv to arrive in any order could be added later)
            else:
                raise ValueError("V must come after Q and K")

        # rename tensor by replacing old keys with new ones
        for old_key, new_key in self.replace_keys.items():
            tensor_name = tensor_name.replace(old_key, new_key)

        return tensor_name, tensor


#============================= PROCESS TENSORS =============================#


def build_safetensors(output_safetensors_path: Path | str,
                      input_rawtensor_file   : IO[bytes],
                      *,
                      header   : SafetensorsHeader,
                      alignment: int = 64,
                      progress: ProgressBar | None = None
                      ):
    """
    Combines the metadata header and the raw binary tensor data into a valid
    single .safetensors file.

    Args:
        output_safetensors_path : Path where the final .safetensors file will be saved.
        input_rawtensor_file    : An open file-like object (in rb or w+b mode) containing
                                  raw tensor data generated with `write_rawtensor_file(..)`.
        header                  : Dictionary containing the metadata of each tensors.
        alignment               : Byte alignment required for the start of the data buffer.
        progress                : An optional progress bar object to track the
                                  writing progress.
    """
    HEADER_START_OFFSET = 8
    COPY_CHUNK_SIZE = 1024 * 1024

    # convert the header to a clean JSON string
    header_str   = json.dumps(header, separators=(",", ":"))
    header_bytes = header_str.encode("utf-8")

    # apply necessary padding to the header
    header_end = HEADER_START_OFFSET + len(header_bytes)
    remainder  = header_end % alignment
    if remainder > 0:
        header_bytes = header_bytes + (b" " * (alignment - remainder))

    # the total size of the header in Little-Endian (uint64)
    header_size = len(header_bytes).to_bytes(8, byteorder="little", signed=False)

    # write the final file by combining everything sequentially
    with open(output_safetensors_path, "wb") as f_out:
        f_out.write(header_size)
        f_out.write(header_bytes)

        # calculate total size for progress tracking
        total_size   = input_rawtensor_file.seek(0, 2)
        written_size = 0
        input_rawtensor_file.seek(0)
        while chunk := input_rawtensor_file.read(COPY_CHUNK_SIZE):
            f_out.write(chunk)
            written_size += len(chunk)
            if progress is not None:
                progress.update(written_size / total_size)


def write_rawtensor_file(output_rawtensor_file: IO[bytes],
                         input_file_paths     : list[Path | str],
                         *,
                         cast_to      : str,
                         tensor_mapper: Callable    | None = None,
                         progress     : ProgressBar | None = None
                         ) -> SafetensorsHeader:
    """
    Process tensors, cast them, and prepare binary data and header metadata for
    future safetensors creation.

    Args:
        output_rawtensor_file: A binary file object where the processed tensor
                               data will be written.
        input_file_paths     : A list of paths to the input source files.
        cast_to              : The target data type string (e.g., "f32", "f16", "bf16").
        tensor_mapper        : An optional callable that takes (tensor_name, tensor)
                               as input and returns a tuple of (new_name, new_tensor),
                               used for custom transformations or filtering.
        progress             : An optional progress bar object to track the
                               processing status.

    Returns:
        A dictionary (SafetensorsHeader) containing the metadata header structured
        for safetensors compatibility, mapping tensor names to their respective
        dtype, shape, and byte offsets.
    """
    cast_to = cast_to.upper()
    if   cast_to in ("F32","FP32","FLOAT32"): st_dtype, dtype = "F32" , np.float32
    elif cast_to in ("F16","FP16","FLOAT16"): st_dtype, dtype = "F16" , np.float16
    elif cast_to in ("BF16")                : st_dtype, dtype = "BF16", ml_dtypes.bfloat16
    else:
        raise ValueError(f"Invalid cast_to value: {cast_to}")

    # start from offset 0 as it is assumed that the raw binary information
    # will be aligned when the final .safetensors is created
    current_offset = 0
    header: SafetensorsHeader = {}

    # validate that all input files exist before starting the process
    for input_file in input_file_paths:
        if not os.path.exists(input_file):
            raise FileNotFoundError(f"File {input_file} does not exist.")

    for input_file in input_file_paths:
        with safe_open(input_file, framework="np", device="cpu") as f_in:
            keys  = f_in.keys()
            total = len(keys)

            for i, tensor_name in enumerate(keys):
                tensor : np.ndarray | None = f_in.get_tensor(tensor_name)
                if tensor is None:
                    continue

                # apply mapping/transformation to tensor
                if tensor_mapper is not None:
                    tensor_name, tensor = tensor_mapper(tensor_name, tensor)
                if not tensor_name or tensor is None:
                    continue

                # convert tensor to raw bytes
                tensor       = tensor.astype(np.dtype(dtype))
                tensor_bytes = tensor.tobytes()
                st_start     = current_offset
                st_end       = st_start + len(tensor_bytes)
                st_shape     = list(tensor.shape)

                # record metadata for the header
                header[tensor_name] = {
                    "dtype"       : st_dtype,
                    "shape"       : st_shape,
                    "data_offsets": [st_start, st_end]
                }

                # write raw bytes
                output_rawtensor_file.write(tensor_bytes)
                current_offset = st_end

                if progress is not None:
                    progress.update((i + 1) / total)

    return header


#===========================================================================#
#////////////////////////////////// MAIN ///////////////////////////////////#
#===========================================================================#

def main(args=None, parent_script=None):
    """
    Main entry point for the script.
    Args:
        args          (optional): List of arguments to parse. Default is None, which will use the command line arguments.
        parent_script (optional): The name of the calling script if any. Used for customizing help output.
    """
    prog = None
    if parent_script:
        prog = parent_script + " " + os.path.basename(__file__).split('.')

    parser = argparse.ArgumentParser(
        prog            = prog,
        description     = "Convert and merge original diffusion model weights to ComfyUI format.",
        formatter_class = argparse.RawTextHelpFormatter,
    )

    parser.add_argument('input_files', nargs='+', metavar='INPUT', help="One or more input safetensors files to process.")
    parser.add_argument('-o', '--output', default='z_image_turbo.safetensors', help="Output safetensors file path.")
    parser.add_argument('-l', '--low-ram', action="store_true", help="Write temporary data to disk instead of RAM, useful for low-memory environments.")
    # mutually exclusive group for precision arguments
    precision_group = parser.add_mutually_exclusive_group()
    precision_group.add_argument('--bf16', action='store_const', const='BF16', dest='dtype', help="Set output precision to BF16 (default).")
    precision_group.add_argument('--fp16', action='store_const', const='FP16', dest='dtype', help="Set output precision to F16.")
    precision_group.add_argument('--fp32', action='store_const', const='FP32', dest='dtype', help="Set output precision to F32.")
    parser.set_defaults(dtype='BF16')
    parsed_args = parser.parse_args(args=args)

    # determine target dtype
    target_dtype = parsed_args.dtype
    message(f"Target data type: {target_dtype}")

    # build path to the output safetensors file
    output_path  = Path(parsed_args.output)
    if not output_path.suffix:
        output_path = output_path.with_suffix(".safetensors")
    new_filename = f"{output_path.stem}_{target_dtype.lower()}{output_path.suffix}"
    output_path  = output_path.with_name(new_filename)

    # prepare the temporary file for in-memory or disk based on --low-ram argument
    if parsed_args.low_ram:
        message("Using disk-based temporary file for low RAM mode.")
        tmp_context = tempfile.TemporaryFile(dir=output_path.parent)
    else:
        message("Using in-memory buffer for temporary data.")
        tmp_context = io.BytesIO()

    with tmp_context as tmp_rawtensor_file:

        progress_bar = ProgressBar()
        safetensor_header = write_rawtensor_file(
                                    tmp_rawtensor_file,
                                    parsed_args.input_files,
                                    cast_to       = target_dtype,
                                    tensor_mapper = TensorMapper(),
                                    progress      = progress_bar)
        tmp_rawtensor_file.seek(0)
        progress_bar = ProgressBar()
        build_safetensors(output_path,
                          input_rawtensor_file = tmp_rawtensor_file,
                          header               = safetensor_header,
                          alignment            = 64,
                          progress             = progress_bar
                          )


if __name__ == "__main__":
    main()
