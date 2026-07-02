"""
File    : convmodel.py
Purpose : Script to convert the original Z-Image model to a format compatible with ComfyUI
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
    from numpy import ndarray
    from typing import Callable
    import ml_dtypes
    from safetensors import safe_open



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


def info(message: str, padding: int = 0, file=sys.stderr) -> None:
    """Displays an informational message to the error stream.
    """
    print(f"{{{' '*padding}}}{CYAN}\u24d8 {message}{RESET}", file=file)


def warning(message: str, *info_messages: str, padding: int = 0, file=sys.stderr) -> None:
    """Displays a warning message to the standard error stream.
    """
    print(f"{{{' '*padding}}}{CYAN}[{YELLOW}WARNING{CYAN}]{DKYELLOW} {message}{RESET}", file=file)
    for info_message in info_messages:
        info(info_message, padding=padding, file=file)


def error(message: str, *info_messages: str, padding: int = 0, file=sys.stderr) -> None:
    """Displays an error message to the standard error stream.
    """
    print(f"{{{' '*padding}}}{DKRED}[{RED}ERROR!{DKRED}]{DKYELLOW} {message}{RESET}", file=file)
    for info_message in info_messages:
        info(info_message, padding=padding, file=file)


def fatal_error(message: str, *info_messages: str, padding: int = 0, file=sys.stderr) -> None:
    """Displays a fatal error message to the standard error stream and exits with status code 1.
    """
    error(message, *info_messages, padding=padding, file=file)
    sys.exit(1)


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
                      input_tensor_file      : IO[bytes],
                      *,
                      header   : dict[str, dict],
                      alignment: int = 64,
                      ):
    """
    Combines the JSON metadata header and the raw binary tensor data into a 
    valid single .safetensors file.

    Args:
        output_safetensors_path : Path where the final .safetensors file will be saved.
        input_tensor_file       : An open file-like object (in 'rb' or 'w+b' mode) containing the tensors.
        header                  : The dictionary containing the metadata of the tensors.
        alignment               : Byte alignment required for the start of the data buffer.
    """
    HEADER_START_OFFSET = 8

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
        while chunk := input_tensor_file.read(64 * 1024):  # chunks of 64KB
            f_out.write(chunk)


def write_binary_tensor_file(output_tensor_rawfile: IO[bytes],
                             input_file_paths     : list[Path | str],
                             *,
                             cast_to    : str,
                             transform  : Callable    | None = None,
                             progress   : ProgressBar | None = None
                             ) -> dict[str, dict]:
    """
    Process tensors, cast them to bfloat16, and prepare binary data and header metadata.

    Args:
        output_file : Path to the output binary file.
        input_files : List of paths to input .safetensors files.
        cast_to     : Target format (e.g., 'bf16').
        progress    : Optional progress tracker object.
    Returns:
        A dictionary containing the metadata header for safetensors.
    """
    if cast_to.lower() != "bf16":
        raise ValueError("Only bf16 is supported for now.")

    # start from offset 0 as it is assumed that the raw binary information
    # will be aligned when the final .safetensors is created
    current_offset = 0
    header = {}

    # validate that all input files exist before starting the process
    for input_file in input_file_paths:
        if not os.path.exists(input_file):
            raise FileNotFoundError(f"File {input_file} does not exist.")

    for input_file in input_file_paths:
        with safe_open(input_file, framework="np", device="cpu") as f_in:
            keys  = f_in.keys()
            total = len(keys)

            for i, tensor_name in enumerate(keys):

                # read tensor and apply transformation
                tensor = f_in.get_tensor(tensor_name)
                if transform is not None:
                    tensor_name, tensor = transform(tensor_name, tensor)
                if not tensor_name or tensor is None:
                    continue

                # convert tensor to raw bytes
                tensor       = tensor.astype(np.dtype(ml_dtypes.bfloat16))
                tensor_bytes = tensor.tobytes()
                start        = current_offset
                end          = start + len(tensor_bytes)

                # record metadata for the header
                header[tensor_name] = {
                    "dtype"       : "BF16",
                    "shape"       : list(tensor.shape), #< list to ensure JSON compatibility
                    "data_offsets": [start, end]
                }

                # write raw bytes
                output_tensor_rawfile.write(tensor_bytes)
                current_offset = end

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
    parser.add_argument('-o', '--output', default='output.safetensors', help="Output safetensors file path.")
    parser.add_argument('-l', '--low-ram', action="store_true", help="Write temporary data to disk instead of RAM, useful for low-memory environments.")
    parser.add_argument('input_files', nargs='+', metavar='INPUT', help="One or more input safetensors files to process.")
    parsed_args = parser.parse_args(args=args)

    # determine target dtype from the output filename
    output_path = parsed_args.output
    cast_to = np.float16
    if "fp16" in output_path:
        cast_to = np.float16
    elif "bf16" in output_path:
        cast_to = ml_dtypes.bfloat16

    # build path to the output safetensors file (and the temporary file)
    output_path = Path(parsed_args.output)
    if output_path.suffix != ".safetensors":
        output_path = output_path.with_suffix(".safetensors")

    if parsed_args.low_ram: temp_context = tempfile.TemporaryFile(dir=output_path.parent)
    else:                   temp_context = io.BytesIO()


    with temp_context as temp_file:

        progress_bar = ProgressBar()
        safetensor_header = write_binary_tensor_file(
                                    temp_file,
                                    parsed_args.input_files,
                                    cast_to="BF16",
                                    transform=TensorMapper(),
                                    progress=progress_bar)

        temp_file.seek(0)
        build_safetensors(output_path, temp_file,
                        header    = safetensor_header,
                        alignment = 64,
                        )




if __name__ == "__main__":
    main()
