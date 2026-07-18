"""
File    : z2comfy.py
Purpose : Script to convert the original Z-Image checkpoint to a format compatible with ComfyUI
Author  : Martin Rizzo | <martinrizzo@gmail.com>
Date    : Jun 26, 2026
Repo    : https://github.com/martin-rizzo/ComfyUI-ZImagePowerNodes
License : MIT
- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                               ZImageCkptTools
          CLI tools for manipulating and verifying Z-Image checkpoints.
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _
"""
import gc
import io
import os
import sys
import copy
import json
import base64
import tempfile
import argparse
from functools import lru_cache
from datetime import datetime, timezone
from pathlib import Path
from typing import Final, IO, NamedTuple, Literal, cast
SCRIPT_NAME = Path(__file__).stem
if __name__ == '__main__' and ("-h" not in sys.argv and "--help" not in sys.argv):

    # Modules that are not available in the standard library must be imported here
    import ml_dtypes
    import numpy as np
    import numpy.typing as npt
    from numpy.typing import NDArray
    from typing      import Callable, Any
    from safetensors import safe_open as safetensors_open

    # Constant to avoid division by zero in Float32 operations.
    # Equivalent to the smallest representable normalized positive number.
    FP32_EPSILON: Final = np.finfo(np.float32).tiny

    # Mapping from numpy data types to their corresponding type identifier
    # used in the safetensors file format
    SAFETENSORS_DTYPES: Final = {
        np.dtype(np.float32): "F32" ,
        np.dtype(np.float16): "F16" ,
        np.dtype(np.float64): "F64" ,
        np.dtype(np.uint8  ): "U8"  ,
        np.dtype(np.int8   ): "I8"  ,
        np.dtype(np.uint16 ): "U16" ,
        np.dtype(np.int16  ): "I16" ,
        np.dtype(np.int32  ): "I32" ,
        np.dtype(np.int64  ): "I64" ,
        np.dtype(np.bool_  ): "BOOL",
        np.dtype(np.uint32 ): "U32" ,
        np.dtype(np.uint64 ): "U64" ,
        np.dtype(ml_dtypes.bfloat16     ): "BF16"   ,
        np.dtype(ml_dtypes.float8_e4m3fn): "F8_E4M3",
        np.dtype(ml_dtypes.float8_e5m2  ): "F8_E5M2",
    }

    # Size in bytes for each supported safetensors data type identifier
    SAFETENSORS_DTYPE_SIZES: Final = {
        "F64": 8, "I64": 8, "U64": 8,
        "F32": 4, "I32": 4, "U32": 4,
        "F16": 2, "BF16": 2, "I16": 2, "U16": 2,
        "F8_E4M3": 1, "F8_E5M2": 1, "I8": 1, "U8": 1, "BOOL": 1
    }

    # Supported output formats.
    TargetFormat = Literal[ "FP32", "FP16", "FP16E", "BF16", "BF16E", "FP8SCALED", "FP8SCALED_E5M2", "INT8CONVROT", "INT4CONVROT" ]

    # Subset of formats that represent high-precision floating point types.
    PrecisionFormat = Literal[ "FP32", "FP16", "FP16E", "BF16", "BF16E" ]
    PRECISION_FORMAT_VALUES =( "FP32", "FP16", "FP16E", "BF16", "BF16E" )

    # Information associated with each output format.
    class TargetFormatInfo(NamedTuple):
        quantized : bool
        rotated   : bool
        file_tag  : str
        dtype     : np.dtype[Any]

    # Properties for each output format, including whether it requires
    # to be quantized, rotated, and a corresponding file tag.
    TARGET_FORMAT_INFO : dict[TargetFormat, TargetFormatInfo] = {
        # TargetFormat  :TargetFormatInfo(quantized, rotated, file_tag       , dtype                             ),
        "FP32"          :TargetFormatInfo(False    , False  , "fp32"         , np.dtype(np.float32)              ),
        "FP16"          :TargetFormatInfo(False    , False  , "fp16"         , np.dtype(np.float16)              ),
        "FP16E"         :TargetFormatInfo(False    , False  , "fp16e"        , np.dtype(np.float16)              ),
        "BF16"          :TargetFormatInfo(False    , False  , "bf16"         , np.dtype(ml_dtypes.bfloat16)      ),
        "BF16E"         :TargetFormatInfo(False    , False  , "bf16e"        , np.dtype(ml_dtypes.bfloat16)      ),
        "FP8SCALED"     :TargetFormatInfo(True     , False  , "fp8scaled"    , np.dtype(ml_dtypes.float8_e4m3fn) ),
        "FP8SCALED_E5M2":TargetFormatInfo(True     , False  , "fp8e5m2scaled", np.dtype(ml_dtypes.float8_e5m2)   ),
        "INT8CONVROT"   :TargetFormatInfo(True     , True   , "int8_convrot" , np.dtype(np.int8)                 ),
        "INT4CONVROT"   :TargetFormatInfo(True     , True   , "int4_convrot" , np.dtype(ml_dtypes.int4)          ),
    }

    # Checkpoint metadata for Z-Image models
    ZIMAGE_METADATA = {
        "title"       : "Z-Image/Z-Image-Turbo",
        "author"      : "Alibaba Tongyi Lab",
        "license"     : "Apache-2.0",
        "description" : (
            "Z-Image is a 6B parameter image generation model based on a "
            "Scalable Single-Stream DiT (S3-DiT) architecture. Optimized to "
            "generate high-quality images, it stands out for its excellent "
            "bilingual (English and Chinese) text rendering capabilities."
        ),
        "architecture": "z-image-v1",
        "tags"        : "Image Generation, S3-DiT, Bilingual, English, Chinese",
        "resolution"  : "1024x1024",
    }

    # Checkpoint metadata for Qwen3 models
    QWEN3_4B_METADATA = {
        "title"       : "Qwen3-4B",
        "author"      : "Alibaba Cloud Qwen Team",
        "license"     : "Apache-2.0",
        "description" : (
            "Qwen3-4B is a 4.0 billion parameter dense large language model "
            "incorporating advanced dual-mode reasoning capabilities. It features "
            "hybrid thinking modes to dynamically switch between high-precision "
            "logical reasoning and efficient dialogue generation across 119 languages."
        ),
        "architecture": "qwen3",
        "tags"        : "LLM, Text Generation, Multilingual, GQA",
        "resolution"  : None,
    }

# Safetensors Header Structure
# each tensor name maps to a dictionary containing its metadata
type SafetensorsHeader = dict[str, dict[str, Any]]

# ConvRot configuration
# Must be a power of 4 for Regular Hadamard (e.g. 16, 64, 256)
CONVROT_GROUP_SIZE = 256

# Size of the data chunk to write to disk
# (the larger the chunk, the faster the writing process but more system memory is required)
WRITE_DATA_CHUNK_SIZE = 256 * (1024 * 1024)


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
        print(f"{pad_spaces} {GREEN}>{RESET} {msg}", end="", file=file)
        for extra in extra_messages:
            print(f" {extra}", end="", file=file)
    print(file=file)


def warning(msg: str, *info_messages: str, padding: int = 0, file=sys.stderr) -> None:
    """Displays a warning message and its info messages with custom padding."""
    pad_spaces = " " * padding
    print(f"{pad_spaces} {CYAN}[{YELLOW}WARNING{CYAN}]{YELLOW} {msg}{RESET}", file=file)
    for info_message in info_messages:
        print(f"{pad_spaces}   {CYAN}\xF0\x9F\x9B\x88 {info_message}{RESET}", file=file)


def error(msg: str, *info_messages: str, padding: int = 0, file=sys.stderr) -> None:
    """Displays an error message and its info messages with custom padding."""
    pad_spaces = " " * padding
    print(f"{pad_spaces} {CYAN}[{RED}ERROR{CYAN}]{RED} {msg}{RESET}", file=file)
    for info_message in info_messages:
        print(f"{pad_spaces}   {CYAN}\xF0\x9F\x9B\x88 {info_message}{RESET}", file=file)



#================================= HELPERS =================================#

def copy_raw_data(f_out   : IO[bytes],
                  f_source: IO[bytes],
                  *,
                  source_offset: int,
                  byte_count   : int,
                  chunk_size   : int | None = None) -> None:
    """
    Copy a specific number of bytes from a source file to a destination file.

    Args:
        f_out        : The destination file object.
        f_source     : The source file object.
        source_offset: The starting position in the source file to begin copying from.
        byte_count   : The total number of bytes to copy.
        chunk_size (optional): The size of each read/write chunk. If None, copies all bytes in one operation.
    """
    f_source.seek(source_offset)

    # copy all bytes in one go
    if chunk_size is None:
        chunk = f_source.read(byte_count)
        if len(chunk) < byte_count: raise EOFError("Unexpected end of file while reading data from temporary file.")
        f_out.write(chunk)
        return

    # copy in blocks (buffered)
    bytes_left = byte_count
    while bytes_left > 0:
        to_read = min(chunk_size, bytes_left)
        chunk   = f_source.read( to_read )
        if len(chunk) < to_read: raise EOFError("Unexpected end of file while reading a block of data from temporary file.")
        f_out.write(chunk)
        bytes_left -= len(chunk)


def create_safetensors_header(*,
                              title         : str | None = None,
                              author        : str | None = None,
                              description   : str | None = None,
                              date          : str | None = None,
                              architecture  : str | None = None,
                              tags          : str | None = None,
                              resolution    : str | None = None,
                              thumbnail_path: str | None = None,
                              implementation: str | None = None,
                              license       : str | None = None,
                              spec_version  : str = "1.0.0"
    ) -> dict[str, Any]:
    """
    Generate a compliant metadata dictionary for a .safetensors header.

    Args:
        title          : Unique identifier for the model.
        author         : Creator of the model.
        description    : Detailed information about the model.
        date           : Creation date (ISO-8601). Defaults to current UTC time if None.
        architecture   : Specific model architecture ID.
        tags           : Comma-separated category labels.
        resolution     : Base resolution for image generation.
        thumbnail_path : Path to JPEG image for base64 encoding.
        implementation : The codebase implementation.
        license        : License terms or link.
        spec_version   : Version of the ModelSpec standard (default: '1.0.0').

    Returns:
        A dictionary containing the '__metadata__' key.
    """
    if isinstance(date,str):
        if date == "*" or date.lower() == "now":
            date = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    fields = {
        "modelspec.sai_model_spec": spec_version,
        "modelspec.implementation": implementation,
        "modelspec.title"         : title,
        "modelspec.author"        : author,
        "modelspec.description"   : description,
        "modelspec.date"          : date,
        "modelspec.architecture"  : architecture,
        "modelspec.tags"          : tags,
        "modelspec.resolution"    : resolution,
        "modelspec.license"       : license
    }

    # filter out `None` values
    filtered_metadata = {k: str(v) for k, v in fields.items() if v is not None}

    # process thumbnail
    if thumbnail_path and os.path.exists(thumbnail_path):
        with open(thumbnail_path, "rb") as image_file:
            encoded = base64.b64encode(image_file.read()).decode("utf-8")
            filtered_metadata["modelspec.thumbnail"] = f"data:image/jpeg;base64,{encoded}"

    return {"__metadata__": filtered_metadata}


def sort_safetensors_header(safetensor_header: dict) -> dict:
    """
    Sorts the header of a safetensors file and recalculates data offsets.

    This function reorganizes tensors within a safetensors header to
    optimize alignment by placing larger data types first.
    """
    # create a deep copy to avoid mutating the original header object
    src_header = copy.deepcopy(safetensor_header)
    new_header = {}

    # preserve global metadata if it exists in the source header
    if "__metadata__" in src_header:
        new_header["__metadata__"] = src_header.pop("__metadata__")

    def get_order(tensor_name: str, tensor_info: dict[str, Any]):
        """Helper to determine the sorting priority of a tensor."""
        dtype = tensor_info.get("dtype", "F32")
        # - primary sort  : dtype size (descending)
        # - secondary sort: dtype name (ascending)
        # - tertiary sort : tensor name (ascending)
        return (-SAFETENSORS_DTYPE_SIZES.get(dtype, 4), dtype, tensor_name)

    # sort the items based on the defined order criteria
    sorted_tensors = sorted(src_header.items(),
                            key = lambda item: get_order(item[0], item[1]))

    # reconstruct the header and recalculate offsets sequentially starting from 0
    current_offset = 0
    for tensor_name, tensor_info in sorted_tensors:
        old_offsets = tensor_info["data_offsets"]
        tensor_size_bytes = old_offsets[1] - old_offsets[0]
        start = current_offset
        end   = start + tensor_size_bytes
        new_header[tensor_name] = {
            "dtype"       : tensor_info["dtype"],
            "shape"       : tensor_info["shape"],
            "data_offsets": [start, end]
        }
        current_offset = end

    return new_header


def detect_model_architecture(input_files: list[Path]) -> str:
    """Detect the model architecture by tracking unique identified suffixes.
    Args:
        input_files: A list of file paths pointing to the model safetensors files.
    Returns:
        A string indicating the detected architecture: "z-image", "qwen3-4b", or "unknown".
    """
    # known suffix sets for each architecture
    Z_IMAGE_SUFFIXES = {"x_pad_token", "cap_pad_token", "layers.27.feed_forward.w2.weight"}
    QWEN_SUFFIXES    = {"model.embed_tokens.weight", "layers.0.mlp.down_proj.weight", "layers.29.self_attn.q_proj.weight"}

    found_z_suffixes, found_qwen_suffixes = set(), set()
    for file_path in input_files:
        with safetensors_open(file_path, framework="np", device="cpu") as safetensors_file:
            for tensor_name in safetensors_file.keys():
                for suffix in Z_IMAGE_SUFFIXES:
                    if tensor_name.endswith(suffix): found_z_suffixes.add(suffix)

                for suffix in QWEN_SUFFIXES:
                    if tensor_name.endswith(suffix): found_qwen_suffixes.add(suffix)

    # check if all required suffixes were found
    if found_z_suffixes    == Z_IMAGE_SUFFIXES: return "z-image"
    if found_qwen_suffixes == QWEN_SUFFIXES   : return "qwen3-4b"
    return "unknown"


def build_metadata_tensor(**metadata: Any) -> np.ndarray:
    """Construct a UINT8 numpy tensor containing serialized metadata for ComfyUI.

    This function acts as a factory, taking arbitrary keyword arguments,
    serializing them into JSON, and returning them as a byte-based tensor
    suitable for storage in ComfyUI or similar frameworks.
    Args:
        **metadata : Arbitrary keyword arguments that will be serialized
                     into the resulting tensor.
    Returns:
        A 1D numpy array of type uint8 containing the encoded metadata.
    """
    byte_data = json.dumps(metadata).encode("utf-8")
    return np.frombuffer(byte_data, dtype=np.uint8)


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

class ZImageTensorMapper:
    def __init__(self, *, aggressive_quantization: bool = False):

            if aggressive_quantization:
                unquantizables = [
                    'cap_pad_token', 'cap_embedder', 't_embedder', 'x_embedder', 'x_pad_token',
                ]
            else:
                unquantizables = [
                    'cap_pad_token', 'cap_embedder', 't_embedder', 'x_embedder', 'x_pad_token',
                    'context_refiner', 'final_layer', 'noise_refiner', 'adaLN',
                    'layers.0.',
                ]


            self.replace_keys = {
                "all_final_layer.2-1."      : "final_layer.",
                "all_x_embedder.2-1."       : "x_embedder.",
                ".attention.to_out.0.bias"  : ".attention.out.bias",
                ".attention.norm_k.weight"  : ".attention.k_norm.weight",
                ".attention.norm_q.weight"  : ".attention.q_norm.weight",
                ".attention.to_out.0.weight": ".attention.out.weight"
            }
            self.unquantizables = unquantizables
            self.qkv_buffers = {}

    def __call__(self, tensor_name: str, tensor: np.ndarray) -> tuple[str|None, np.ndarray|None, bool]:

        # omit tensors that cause conflicts or are unnecessary
        if tensor_name.endswith(".attention.to_out.0.bias"):
            return None, None, False

        # extract the layer prefix (e.g., "model.layers.0")
        layer_prefix = tensor_name.rsplit(".attention.", 1)[0] if ".attention." in tensor_name else ""

        # logic for accumulating QKV tensors
        if tensor_name.endswith(".attention.to_k.weight"):
            if layer_prefix not in self.qkv_buffers: self.qkv_buffers[layer_prefix] = {}
            self.qkv_buffers[layer_prefix]['k'] = tensor
            return None, None, False

        if tensor_name.endswith(".attention.to_q.weight"):
            if layer_prefix not in self.qkv_buffers: self.qkv_buffers[layer_prefix] = {}
            self.qkv_buffers[layer_prefix]['q'] = tensor
            return None, None, False

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

        is_rotatable = all(unquant not in tensor_name for unquant in self.unquantizables)
        if tensor.ndim != 2:
            is_rotatable = False
        return tensor_name, tensor, is_rotatable


class Qwen3TensorMapper:
    def __init__(self):
            self.unrotatable = [ 'embed_tokens' ]

    def __call__(self, tensor_name: str, tensor: np.ndarray) -> tuple[str|None, np.ndarray|None, bool]:

        ## omit tensors that cause conflicts or are unnecessary
        #if tensor_name.endswith("???"):
        #    return None, None, False

        # rename tensor by replacing old keys with new ones
        #for old_key, new_key in self.replace_keys.items():
        #    tensor_name = tensor_name.replace(old_key, new_key)

        is_rotatable = all(unquant not in tensor_name for unquant in self.unrotatable)
        if tensor.ndim != 2:
            is_rotatable = False
        return tensor_name, tensor, is_rotatable


#========================= FP8 SCALED QUANTIZATION =========================#

def quantize_fp8_scaled(tensor: np.ndarray,
                        *,
                        target_format        : TargetFormat,
                        scales_format        : PrecisionFormat,
                        stochastic_generator : np.random.Generator,
                        ) -> tuple[np.ndarray, np.ndarray]:
    """Quantizes a tensor to FP8 format using a scale factor.
    Args:
        tensor               : Input numpy array to quantize. (usually FP32/FP16).
        target_format        : The specific FP8 variant to use (FP8SCALED or FP8SCALED_E5M2).
        scales_format        : The precision format used for storing the resulting scale factor.
        stochastic_generator : The `np.random.Generator` used if stochastic rounding is required.
    Returns:
        A tuple containing:
            - quantized_fp8: The resulting tensor quantized to the target FP8 dtype.
            - scale        : The scale factor (represented in the requested precision format)
                             required to dequantize the values.
    """
    if target_format not in ("FP8SCALED", "FP8SCALED_E5M2"):
        raise ValueError(f"Invalid `target_format` for fp8 quantization: {target_format}")
    if scales_format not in PRECISION_FORMAT_VALUES:
        raise ValueError(f"Invalid `scales_format` for fp8 quantization: {scales_format}")

    # get FP8 specific properties
    _, _, _, fp8_dtype = TARGET_FORMAT_INFO[target_format]
    fp8_max = ml_dtypes.finfo(fp8_dtype).max

    # convert to float32 for high-precision intermediate calculations
    tensor_fp32 = tensor.astype(np.float32, copy=False)
    max_abs = np.max(np.abs(tensor_fp32))

    # calculate optimal scale, avoiding division by zero
    scale = 1.0 if max_abs == 0 else fp8_max / max_abs

    # scale the tensor and apply clipping to prevent overflow
    scaled_tensor = np.clip(tensor_fp32 * scale, -fp8_max, fp8_max)
    quantized_fp8 = scaled_tensor.astype(fp8_dtype)

    # determine scale storage dtype
    _, _, _, scales_dtype = TARGET_FORMAT_INFO[scales_format]

    # return the inverse scale so that: weight_fp32 = weight_fp8 * inverse_scale
    scale_inv = np.array([1.0 / scale], dtype=scales_dtype)

    return quantized_fp8, scale_inv


def preload_fp8_scales(path: Path) -> dict[str, str]:
    """Parses a fp8scales file into a dictionary.
    Args:
        path : A `Path` object pointing to the scale configuration file.
    Returns:
        A dictionary where keys represent tensor names and values represent
        their corresponding scale factors.
    """
    scales = {}
    with path.open('r', encoding='utf-8') as file:

        for line_num, line in enumerate(file, start=1):
            clean_line = line.strip()

            # skip empty lines and comments starting with '#'
            # and validate line format
            if not clean_line or clean_line.startswith('#'):
                continue
            if not ':' in clean_line:
                raise ValueError(f"Invalid line format in line {line_num}")

            key, _, value = clean_line.partition(':')
            scales[key.strip()] = value.strip()

    return scales


def read_fp8_preloaded_scale(tensor_name: str,
                             fp8_preloaded_scales : dict[str,str] | None,
                             *,
                             format: PrecisionFormat
                             ) -> np.ndarray:
    """Get the `.scale_input` value for a specific tensor.
    Args:
        tensor_name         : The identifier of the tensor for which to retrieve the scale.
        fp8_preloaded_scales: A preloaded dictionary containing the mapping of tensor names to scale values.
        format              : The target precision format for the scaling value.
    Returns:
        A NumPy array containing the scale factor as a float, cast to the dtype
        required by the precision format.
    """
    _, _, _, dtype = TARGET_FORMAT_INFO[format]
    scale_float = 1.0

    if fp8_preloaded_scales:
        value = fp8_preloaded_scales.get(tensor_name)
        if value is None:
            warning(f"Missing scale for {tensor_name}. Defaulting to 1.0")
        else:
            try:
                scale_float = float(value)
            except:
                warning(f"Could not convert scale value '{value}' for {tensor_name} to float. Defaulting to 1.0")
                scale_float = 1.0

    return np.array([scale_float], dtype=dtype)



#========================= INT CONVROT QUANTIZATION =========================#

def quantize_int8_convrot(tensor: np.ndarray,
                          *,
                          group_size           : int,
                          scales_format        : PrecisionFormat,
                          scales_search_trials : int,
                          stochastic_generator : np.random.Generator,
                          ) -> tuple[np.ndarray, np.ndarray]:
    """Perform a ConvRot tensor rotation followed by row-wise INT8 quantization.
    Args:
        tensor              : The input tensor to be quantized.
        group_size          : Size of the group for rotation.
        scales_format       : String representing the target dtype for the scales. Supported
                              values are "FP32", "FP16", "BF16", "FP16E", and  "BF16E".
                              Any other value will be treated as FP32.
        scale_search_trials : Number of candidate scale factors to test during optimization
                              to minimize quantization error (MSE).
        stochastic_generator: The `np.random.Generator` used when stochastic rounding is needed.
    Returns:
        A tuple containing:
            - quantized_int8: An INT8 array resulting from the rotation and quantization process.
            - scales        : Array containing per-row scales with shape [..., 1], casted
                              to the specified `scales_format`.
    """
    hadamard_matrix = _build_hadamard_matrix(group_size, dtype="float32")
    rotated_tensor  = _rotate_tensor(tensor, hadamard_matrix, group_size)
    return _quantize_int8_rowwise(rotated_tensor,
                                  scales_format        = scales_format,
                                  scales_search_trials = scales_search_trials,
                                  stochastic_generator = stochastic_generator)


def _quantize_int8_rowwise(tensor: np.ndarray,
                           *,
                           scales_format        : PrecisionFormat,
                           scales_search_trials : int,
                           stochastic_generator : np.random.Generator,
                          ) -> tuple[np.ndarray, np.ndarray]:
    """Quantize a tensor to INT8 with per-row scales, supporting custom scale dtypes.

    Args:
        tensor              : Input array of shape [..., K] where quantization will be performed.
        scales_format       : String representing the target dtype for the scales. Supported
                              values are "FP32", "FP16", "BF16", "FP16E", and  "BF16E".
                              Any other value will be treated as FP32.
        scale_search_trials : Number of candidate scale factors to test during optimization
                              to minimize quantization error (MSE).
        stochastic_generator: The `np.random.Generator` used when stochastic rounding is needed.

    Returns:
        A tuple containing:
            - quantized_int8: An INT8 array with the same shape as the input tensor.
            - scales        : Array containing per-row scales with shape [..., 1], casted
                              to the specified `scales_format`.
    """
    if tensor.dtype != np.float32:
        tensor = tensor.astype(np.float32)

    scales_dtype = TARGET_FORMAT_INFO[scales_format].dtype

    # calculate scales using the absolute maximum value of each ???
    abs_maximum = np.max(np.abs(tensor), axis=-1, keepdims=True)
    scales = np.maximum(abs_maximum / 127.0, FP32_EPSILON)

    # optimize scales when requested by the user
    if scales_search_trials > 1:
        quantized_int8, scales = optimize_quantization_scales(tensor, scales,
                                        scales_dtype = scales_dtype,
                                        quant_dtype  = np.dtype(np.int8),
                                        quant_min    = np.iinfo(np.int8).min,
                                        quant_max    = np.iinfo(np.int8).max,
                                        quant_is_int = True,
                                        num_trials   = scales_search_trials)
    else:
        # if not trial allowed, use naive quantization
        quantized_int8 = (tensor / scales).astype(np.int8)
        scales         = scales.astype(scales_dtype)

    return quantized_int8, scales


def optimize_quantization_scales(tensor: NDArray[np.float32],
                                 scales: NDArray[np.float32],
                                 *,
                                 scales_dtype: np.dtype[Any],
                                 quant_dtype : np.dtype[Any],
                                 quant_min   : float | int,
                                 quant_max   : float | int,
                                 quant_is_int: bool,
                                 num_trials: int = 30) -> tuple[NDArray, NDArray]:
    """
    Find optimal quantization scales for a given tensor by testing multiple scaling factors.

    Args:
        tensor       : Input tensor of shape [R, C] to be quantized.
        scales       : Initial scale factors per row, shape [R, 1].
        scales_dtype : The data type used for the resulting scales.
        quant_dtype  : The data type used for the quantized tensor.
        quant_min    : The minimum value for the quantized range.
        quant_max    : The maximum value for the quantized range.
        quant_is_int : `True` if the quantization is integer, `False` if floating point.
        num_trials   : Number of scale variations to test.

    Returns:
        A tuple containing the quantized tensor and the optimized scales.
    """
    # tensor: [R, C] -> R: Rows, C: Columns
    # scales: [R, 1] -> An initial scale per row

    # define scale options [T, R, 1] where T = num_trials
    steps = np.linspace(0.96, 1.005, num=num_trials, dtype=np.float32)
    scales_options = steps[:, np.newaxis, np.newaxis] * scales[np.newaxis, :, :]
    scales_options = scales_options.astype(scales_dtype).astype(np.float32)

    # expand the original tensor for calculation -> [1, R, C]
    tensor_orig = tensor[np.newaxis, :, :]

    # quantize using ALL possible scales
    raw_quant = tensor_orig / scales_options
    if quant_is_int:
        raw_quant = np.round(raw_quant)
    raw_quant = np.clip(raw_quant, quant_min, quant_max).astype(quant_dtype)

    # dequantize back to float32,
    # approximating the original tensor with its quantization error
    tensor_approx = raw_quant.astype(np.float32) * scales_options

    # destroy scales_options right now to free memory
    del scales_options
    gc.collect()

    # calculate the error (MSE) for each scale option and row: [T, R]
    error_per_row = np.mean((tensor_orig - tensor_approx) ** 2, axis=2)

    # select the best scale per row (index of the minimum error among the T trials)
    best_indices = np.argmin(error_per_row, axis=0)

    ## print the complete array of best indices (for debugging)
    #with np.printoptions(threshold=sys.maxsize):
    #    print("##>> best_indices:", best_indices)
    #    print("##-------------------------------------------------------------")

    best_steps = steps[best_indices]
    best_scales = scales * best_steps[:, np.newaxis]
    best_scales.astype(scales_dtype)

    # quantize using the best scales and return
    raw_quant = tensor / best_scales
    if quant_is_int:
        raw_quant = np.round(raw_quant)
    raw_quant = np.clip(raw_quant, quant_min, quant_max).astype(quant_dtype)

    return raw_quant, best_scales


def _rotate_tensor(tensor      : np.ndarray,
                   h_matrix    : np.ndarray,
                   group_size  : int,
                   ) -> np.ndarray:
    """Rotate a tensor matrix offline: T_rot = T @ H^T.

    For a Linear(in_features, out_features) layer with a tensor shape (out_features, in_features):
    Each row of the tensor is split into groups of group_size and rotated by H^T.

    Args:
        tensor     : The input tensor matrix of shape (out_features, in_features).
        h_matrix   : Normalized Hadamard matrix of shape (group_size, group_size).
        group_size : Group size for block-diagonal rotation. Must divide in_features.

    Returns:
        The rotated tensor with the same shape as the input.
    """
    out_features, in_features = tensor.shape

    if in_features % group_size != 0:
        raise ValueError(
            f"in_features {in_features} is not divisible by group_size {group_size}"
        )

    n_groups = in_features // group_size

    # Reshape (out, in) -> (out, n_groups, group_size)
    tensor_grouped = tensor.reshape(out_features, n_groups, group_size)

    # Apply H^T to each group: (..., group_size) @ (group_size, group_size)
    # Using np.matmul to handle the multiplication over the last dimensions
    h_transpose = h_matrix.T.astype(tensor.dtype)
    rotated_grouped = np.matmul(tensor_grouped, h_transpose)

    # Reshape back to (out, in)
    return rotated_grouped.reshape(out_features, in_features)


@lru_cache(maxsize=8)
def _build_hadamard_matrix(size: int, dtype: str = "float32") -> np.ndarray[Any, np.dtype[Any]]:
    """
    Build a normalized regular orthogonal Hadamard matrix.

    Args:
        size  : The size of the square matrix. Must be a power of 4.
        dtype : String representation of the NumPy data type. This is passed
                as a string to ensure compatibility with the `lru_cache` mechanism.
    Returns:
        A normalized Hadamard matrix of shape (size, size).
    """
    if not isinstance(dtype,str):
        raise TypeError(f"Expected dtype to be a string, got '{type(dtype).__name__}'")

    if size < 4 or (size & (size - 1)) != 0 or (size.bit_length() - 1) % 2 != 0:
        raise ValueError(f"Regular Hadamard size must be a power of 4, got {size}")

    # base hadamard matrix of order 4,
    # in this specific construction, every row and column sums to exactly 2
    H4 = np.array([ [ 1,  1,  1, -1],
                    [ 1,  1, -1,  1],
                    [ 1, -1,  1,  1],
                    [-1,  1,  1,  1]  ], dtype=np.dtype(dtype))
    # iteratively build the full hadamard matrix using the kronecker product
    H, current_size = H4, 4
    while current_size < size:
        H = np.kron(H, H4)
        current_size *= 4

    return H / np.sqrt(size, dtype=np.dtype(dtype))



#============================= PROCESS TENSORS =============================#

def softplus_clamp(tensor, clamp_limit, *, sharpness=1.2):
    """
    Applies a smooth pseudo-clamping operation to a tensor using softplus functions.

    The result is a smooth, fully differentiable approximation of a hard clip.
    It behaves approximately like the identity function in the center of the range,
    but compresses values across boundaries to asymptotically approach the +-clamp_limit.

    Args:
        tensor      : Input tensor to be clamped.
        clamp_limit : Maximum absolute value allowed. Values beyond this limit
                      are smoothly clamped to the boundary values.
        sharpness   : Controls how quickly the function flattens near the
                      clamping boundaries. Higher values create sharper transitions.
                      Default is 1.2.
    Returns:
        Tensor of the same shape as input, with smooth clamping applied.
    """
    # softplus(x) = ln(1 + exp(x))
    def softplus(x):
        return np.log1p(np.exp(np.clip(x, -20, 20)))

    # symmetric combination to bound both negative and positive sides
    tp = tensor + clamp_limit
    tm = tensor - clamp_limit
    return tensor - (softplus(sharpness * tm) - softplus(-sharpness * tp)) / sharpness


def build_safetensors(output_safetensors_path: Path | str,
                      *,
                      output_header        : SafetensorsHeader,
                      sour_rawtensor_header: SafetensorsHeader,
                      sour_rawtensor_file  : IO[bytes],
                      alignment: int = 64,
                      progress : ProgressBar | None = None
                      ):
    """
    Combines the metadata header and the raw binary tensor data into a valid
    single .safetensors file by reordering tensors according to output_header.

    Args:
        output_safetensors_path: Path where the final .safetensors file will be saved.
        output_header          : The final ordered header with its own calculated offsets.
        sour_rawtensor_header  : The original header mapping to the temporary raw file.
        sour_rawtensor_file    : Open IO stream of the temporary raw tensor file.
        alignment              : Byte alignment required for the start of the tensor raw data.
        progress               : An optional progress bar object to track the writing progress.
    """
    HEADER_START_OFFSET = 8

    # convert the header to a clean JSON utf8 string
    output_header_bytes = json.dumps(output_header, separators=(",", ":")).encode("utf-8")

    # apply necessary padding to the header
    remainder = (HEADER_START_OFFSET + len(output_header_bytes)) % alignment
    if remainder > 0:
        output_header_bytes = output_header_bytes + (b" " * (alignment - remainder))

    # the total size of the header in Little-Endian (uint64)
    output_header_size = len(output_header_bytes).to_bytes(8, byteorder="little", signed=False)

    # write the final file by combining everything sequentially
    with open(output_safetensors_path, "wb") as f_out:
        f_out.write(output_header_size)
        f_out.write(output_header_bytes)

        # calculate the number of tensors for the progress bar
        total_count = len(output_header)

        # copy each tensor one by one in the order of the output header
        for index, (tensor_name, tensor_data) in enumerate(output_header.items()):
            if progress is not None:
                progress.update((index+1) / total_count)
            if tensor_name.startswith('__'):
                continue

            # get source and destination offsets
            sour_start, sour_end = sour_rawtensor_header.get(tensor_name, {}).get("data_offsets", (0, 0))
            dest_start, dest_end = tensor_data.get("data_offsets", (0, 0))
            tensor_size = sour_end - sour_start

            # validate the consistency of the tensor size
            if tensor_size < 0:
                raise ValueError(f"Tensor '{tensor_name}' has a negative size. This could be a bug in the offset calculation.")
            if tensor_size != (dest_end - dest_start):
                raise ValueError(
                    f"Tensor '{tensor_name}' has different sizes between source ({tensor_size} bytes) "
                    f"and destination ({dest_end - dest_start} bytes). This is a bug in the offset calculation.")

            # validate the alignment of the output file
            # (the packed header + the relative offset of the tensor must match where the actual pointer is)
            expected_pos = HEADER_START_OFFSET + len(output_header_bytes) + dest_start
            if expected_pos != f_out.tell():
                raise ValueError(
                    f"Desalignment detected when writing '{tensor_name}'. "
                    f"Current file position: {f_out.tell()}, Expected header position: {expected_pos}.")

            # finally, copy the raw tensor data from source to destination
            if tensor_size>0:
                copy_raw_data(f_out, sour_rawtensor_file, source_offset=sour_start, byte_count=tensor_size, chunk_size=WRITE_DATA_CHUNK_SIZE)



def cast_tensor(tensor_name: str,
                tensor     : np.ndarray,
                *,
                cast_to             : str,
                scales_format       : PrecisionFormat,
                scales_search_trials: int,
                stochastic_generator: np.random.Generator,
                fp8_preloaded_scales: dict[str,str] | None = None,
                ) -> dict:
    """
    Cast a tensor to a target dtype, optionally using stochastic rounding.

    Args:
        tensor_name          : The name identifier of the tensor.
        tensor               : The input tensor as a numpy array.
        cast_to              : The target format (FP32, FP16, BF16, FP16E, BF16E, INT8CONVROT).
        scales_format        : String representing the dtype for the scales in quantized types (e.g., INT8CONVROT).
                               Supported "FP16", "BF16", "FP16E", "BF16E". Any other value will be treated as FP32.
        stochastic_generator : A `np.random.Generator` instance used for stochastic rounding.

    Returns:
        A dictionary containing the casted tensor and optional metadata.
    """
    cast_to = cast_to.upper()
    parent  = tensor_name.rsplit('.', 1)[0] if '.' in tensor_name else tensor_name

    if cast_to == "FP32":
        return {
            tensor_name: tensor.astype(np.float32) }

    elif cast_to == "FP16":
        return {
            tensor_name: tensor.astype(np.float16) }

    elif cast_to == "BF16":
        return {
            tensor_name: tensor.astype(np.dtype(ml_dtypes.bfloat16)) }

    elif cast_to == "FP16E": # FP16 with exact stochastic rounding
        tensor = tensor.astype(np.float32, order='C', copy=True)
        tensor_fp16_down = np.nextafter(tensor, -np.inf).astype(np.float16).astype(np.float32)
        tensor_fp16_up   = np.nextafter(tensor_fp16_down, np.inf).astype(np.float16).astype(np.float32)
        eps = tensor_fp16_up - tensor_fp16_down
        fraction = np.divide(
            tensor - tensor_fp16_down, eps,
            out=np.zeros_like(tensor), where=eps != 0
        )
        noise = stochastic_generator.uniform(0.0, 1.0, size=tensor.shape).astype(np.float32)
        rounded_fp32 = np.where(fraction > noise, tensor_fp16_up, tensor_fp16_down)
        return {
            tensor_name: rounded_fp32.astype(np.float16) }

    elif cast_to == "BF16E": # BF16 with binary stochastic rounding
        tensor = tensor.astype(np.float32, order='C', copy=True)
        noise_int       = stochastic_generator.integers(0, 65536, size=tensor.shape, dtype=np.uint32)
        stochastic_bits = (tensor.view(np.uint32) + noise_int) & 0xFFFF0000
        stochastic_fp32 = stochastic_bits.view(np.float32)
        return {
            tensor_name: stochastic_fp32.astype(np.dtype(ml_dtypes.bfloat16)) }

    elif cast_to == "FP8SCALED":
        input_scales_format   = scales_format
        quantized_fp8, scales = quantize_fp8_scaled(tensor, target_format="FP8SCALED", scales_format=scales_format, stochastic_generator=stochastic_generator)
        scale_input           = read_fp8_preloaded_scale(f"{parent}.scale_input", fp8_preloaded_scales, format=input_scales_format)
        return {
            f"{parent}.weight"      : quantized_fp8,
            f"{parent}.scale_weight": scales,
            f"{parent}.scale_input" : scale_input }

    elif cast_to == "INT8CONVROT":
        quantized_int8, scales = quantize_int8_convrot(tensor,
                                                       group_size           = CONVROT_GROUP_SIZE,
                                                       scales_format        = scales_format,
                                                       scales_search_trials = scales_search_trials,
                                                       stochastic_generator = stochastic_generator)
        comfy_quant = build_metadata_tensor( format="int8_tensorwise", convrot=True, convrot_groupsize=CONVROT_GROUP_SIZE )
        return {
            f"{tensor_name}"       : quantized_int8,
            f"{tensor_name}_scale" : scales,
            f"{parent}.comfy_quant": comfy_quant }

    elif cast_to == "INT4CONVROT":
        comfy_quant = build_metadata_tensor( format="convrot_w4a4", convrot_groupsize=CONVROT_GROUP_SIZE )
        raise NotImplementedError(f"Conversion to {cast_to} is not implemented yet")

    else:
        raise ValueError(f"Unknown dtype: {cast_to}")



def convert_safetensors_file(input_safetensors_file: Path,
                             output_rawtensor_file : IO[bytes],
                             *,
                             target_format        : TargetFormat,
                             scales_format        : PrecisionFormat | None     = None,
                             high_precision_format: PrecisionFormat | None     = None,
                             scales_search_trials : int | None                 = None,
                             stochastic_generator : np.random.Generator | None = None,
                             fp8_preloaded_scales : dict[str,str] | None       = None,
                             tensor_mapper        : Callable | None            = None,
                             clamp_limit          : float | None               = None,
                             clamp_sharpness      : float | None               = None,
                             progress             : Any | None                 = None
                             ) -> SafetensorsHeader:
    """
    Process a tensor file, cast it, and append its binary data to the output file.
    Returns the safetensors header for the processed tensors.

    Args:
        output_rawtensor_file : A binary file object where the processed tensor
                                data will be appended.
        input_safetensors_file: Path to the input source file.
        target_format         : The target dtype format.
                                Supported "FP32", "FP16", "FP16E", "BF16", "BF16E", "FP8SCALED", "FP8SCALED_E5M2", "INT8CONVROT", "INT4CONVROT"
        scales_format         : String representing the dtype for the scales in quantized types (e.g., when target is INT8CONVROT).
                                Supported "FP32", "FP16", "FP16E", "BF16", "BF16E". Any other value will be treated as FP32.
        high_precision_format : String representing the high precision dtype format to use when the tensor can't be quantized.
                                Supported "FP32", "FP16", "FP16E", "BF16", "BF16E". Any other value will be treated as FP32.
        stochastic_generator  : A numpy random generator for stochastic rounding.
        tensor_mapper         : Optional callable for custom tensor transformations.
        clamp_limit           : Optional limit for tensor values clamping.
        clamp_sharpness       : Optional sharpness parameter for the clamping curve.
        progress              : Optional progress bar object.

    Returns:
        A dict (SafetensorsHeader) containing the metadata for the processed tensors.
    """
    if not input_safetensors_file.exists():
        raise FileNotFoundError(f"File {input_safetensors_file} does not exist.")

    # get the target format information
    if target_format not in TARGET_FORMAT_INFO:
        raise ValueError(f"Invalid `target_format` value: {target_format}")
    req_quantization, req_rotation, file_tag, quant_dtype = TARGET_FORMAT_INFO[target_format]

    # set format for quantization scales if not provided and validate it
    if scales_format is None:
        scales_format = "FP32"
    if scales_format not in PRECISION_FORMAT_VALUES:
        raise ValueError(f"Invalid `scales_format` value: {scales_format}")

    # set the high precision dtype if not provided and validate it
    if high_precision_format is None:
        high_precision_format = "FP32"
    if high_precision_format not in PRECISION_FORMAT_VALUES:
        raise ValueError(f"Invalid `high_precision_format` value: {high_precision_format}")

    # initialize the stochastic generator if it's not provided
    if stochastic_generator is None:
        stochastic_generator = np.random.default_rng()

    header: SafetensorsHeader = {}
    with safetensors_open(input_safetensors_file, framework="np", device="cpu") as safetensors_file:

        keys  = safetensors_file.keys()
        total = len(keys)
        for i, tensor_name in enumerate(keys):

            # get tensor info and transform it with the given `tensor_mapper`
            tensor         = cast(np.ndarray, safetensors_file.get_tensor(tensor_name))
            is_quantizable = True
            is_rotatable   = True
            if (tensor_mapper is not None) and (tensor is not None):
                tensor_name, tensor, is_quantizable = tensor_mapper(tensor_name, tensor)

            # discard null tensors that may have been generated by the `tensor_mapper`
            if not tensor_name or tensor is None:
                continue

            # clamp tensor if required by the user
            if clamp_limit is not None:
                tensor = softplus_clamp(tensor,
                                        clamp_limit = clamp_limit,
                                        sharpness   = clamp_sharpness if clamp_sharpness is not None else 1.2)

            # determine the target format for the tensor based on its properties
            cast_to = target_format
            if req_quantization:
                cast_to = target_format if is_quantizable else high_precision_format

            # cast tensor to the desired dtype,
            # the result is a dictionary because some casting processes generate multiple tensors for ComfyUI
            out_tensor_dict = cast_tensor(tensor_name, tensor,
                                          cast_to              = cast_to,
                                          scales_format        = scales_format,
                                          scales_search_trials = scales_search_trials or 0,
                                          stochastic_generator = stochastic_generator,
                                          fp8_preloaded_scales = fp8_preloaded_scales)

            # convert casted tensor to bytes and store them in `output_rawtensor_file`
            for out_tensor_name, out_tensor in out_tensor_dict.items():
                if not isinstance(out_tensor, np.ndarray):
                    continue
                out_bytes = out_tensor.tobytes()
                out_start = output_rawtensor_file.tell()
                out_end   = out_start + len(out_bytes)
                out_shape = list(out_tensor.shape)

                # record metadata for the header
                header[out_tensor_name] = {
                    "dtype"       : SAFETENSORS_DTYPES[out_tensor.dtype],
                    "shape"       : out_shape,
                    "data_offsets": [out_start, out_end]
                }

                # write raw bytes
                output_rawtensor_file.write(out_bytes)

            if progress is not None:
                progress.update((i + 1) / total)

    # Hack for telling ComfyUI that the checkpoint is in fp8-scaled
    # (there is a more modern way to do this by using the metadata field of the safetensors header)
    if target_format == "FP8SCALED":
        header["scaled_fp8"] = {
            "dtype"       : "F8_E4M3",
            "shape"       : [0],
            "data_offsets": [output_rawtensor_file.tell(), output_rawtensor_file.tell()]
        }

    return header


def parse_clamp_args(clamp_str        : str,
                     default_sharpness: float = 0.8
                     ) -> tuple[float, float] | tuple[None, None]:
    """
    Parse a clamp argument string in the format "limit" or "limit:sharpness" 
    into a tuple of floats.

    Args:
        clamp_str        : String in format "limit" or "limit:sharpness".
        default_sharpness: The value to use if no sharpness is provided in the string.

    Returns:
        Tuple of (limit, sharpness) as floats, or (None, None) if parsing fails.
    """
    try:
        if ':' in clamp_str:
            limit_str, _, sharpness_str = clamp_str.partition(':')
            limit     = float(limit_str)
            sharpness = float(sharpness_str) if sharpness_str else default_sharpness
        else:
            limit     = float(clamp_str)
            sharpness = default_sharpness
        return limit, sharpness

    except ValueError:
        return None, None



def make_clamping_tag(limit: float | None, sharpness: float | None) -> str:
    """
    Constructs a tag string for clamping parameters to be appended to output filenames.

    Args:
        limit     : The maximum value for clamping.
        sharpness : The sharpness factor of the clamping curve.

    Returns:
        A formatted string tag in the format '_clamp<LIMIT>s<SHARPNESS>'
        or '_clamp<LIMIT>' if sharpness is None. Numeric values are formatted
        by removing the decimal point and keeping one decimal place (3.14 -> 31, 7 -> 70).
    """
    def format_value(value: float) -> str:
        return f"{int(round(value * 10)):02d}"

    if limit is None:
        return ""
    elif sharpness is None:
        return f"_clamp{format_value(limit)}"
    else:
        return f"_clamp{format_value(limit)}sharp{format_value(sharpness)}"


#===========================================================================#
#////////////////////////////////// MAIN ///////////////////////////////////#
#===========================================================================#

def validate_and_collect_safetensors(input_files: list[str | Path]) -> list[Path]:
    """
    Validate input paths and collect safetensors files from a directory or list.

    Args:
        input_files: A list of paths provided by the user via command line,
                     pointing to files or a single directory.
    Returns:
        A list of validated Path objects to safetensors files.
    """
    paths      : list[Path] = [Path(f) for f in input_files]
    valid_files: list[Path] = []
    directory: Path | None = None

    # check if any file is actually a directory
    for file_or_dir in paths:
        if file_or_dir.is_dir():
            if directory is None:
                directory = file_or_dir
            else: raise ValueError("Multiple directories provided. Only one directory is allowed.")

    # if directory is present, ensure it is the only element
    if directory and len(paths) > 1:
        raise ValueError("Cannot specify a directory and individual files simultaneously.")

    # search for "model*.safetensors" and "diffusion*.safetensors"
    # inside directory and inside directory/transformer
    if directory:
        search_paths = [directory, directory / "transformer"]
        patterns     = ["model*.safetensors", "diffusion*.safetensors"]
        for base_path in search_paths:
            if not base_path.exists():
                continue
            for pattern in patterns:
                valid_files.extend(base_path.glob(pattern))
        valid_files.sort()

    # if not a directory, treat all as individual files
    if not directory:
        valid_files = paths

    if not valid_files:
        raise ValueError(f"No valid .safetensors files.")

    return valid_files


def main(parent_args  : list[str] | None = None,
         parent_script: str | None       = None
         ) -> None:
    """
    Main entry point for the CLI tool.
    Args:
        parent_args   : List of arguments to parse or `None` for reading from command line.
        parent_script : The name of the calling script if any.
    """
    parser = argparse.ArgumentParser(
        prog        = f"{parent_script} {SCRIPT_NAME}" if parent_script else SCRIPT_NAME,
        description = (
            "Convert Z-Image checkpoint files into various formats compatible with ComfyUI.\n\n"
            "This utility supports multiple precision formats (FP32, FP16, BF16), integer "
            "quantization (I8CONVROT), and stochastic rounding techniques. "
            "While originally designed for the official Z-Image files "
            "(e.g., https://huggingface.co/Tongyi-MAI/Z-Image-Turbo), "
            "this tool should be compatible with any checkpoint within this model family."
        ),
        formatter_class = argparse.RawTextHelpFormatter,
    )

    parser.add_argument('input_files', nargs='+', metavar='INPUT', help="One or more input safetensors files to process.")
    parser.add_argument('-o', '--output' , help="Output safetensors file path.")
    parser.add_argument('-l', '--low-ram', action="store_true", help="Write temporary data to disk instead of RAM, useful for low-memory environments.")
    sort_group = parser.add_mutually_exclusive_group()
    sort_group.add_argument('--sort', action='store_true', dest='sort_tensors', default=True,
                            help="Enable sorting of tensors by size in the output file (default).")
    sort_group.add_argument('--no-sort', action='store_false', dest='sort_tensors',
                            help="Disable sorting of tensors in the output file.")

    #-- Precision & Quantization Options --------
    precision_main_group = parser.add_argument_group('precision & quantization options')
    precision_group = precision_main_group.add_mutually_exclusive_group()
    precision_group.add_argument('--fp32'       , action='store_const', const='FP32'       , dest='dtype', help="Set output precision to F32.")
    precision_group.add_argument('--fp16'       , action='store_const', const='FP16'       , dest='dtype', help="Set output precision to F16.")
    precision_group.add_argument('--bf16'       , action='store_const', const='BF16'       , dest='dtype', help="Set output precision to BF16 (default).")
    precision_group.add_argument('--fp16e'      , action='store_const', const='FP16E'      , dest='dtype', help="Set output precision to FP16 with stochastic rounding.")
    precision_group.add_argument('--bf16e'      , action='store_const', const='BF16E'      , dest='dtype', help="Set output precision to BF16 with stochastic rounding.")
    precision_group.add_argument('--fp8scaled'  , action='store_const', const='FP8SCALED'  , dest='dtype', help="Set output precision to FP8 with scale factors.")
    precision_group.add_argument('--int8convrot', action='store_const', const='INT8CONVROT', dest='dtype', help="Set output precision to INT8 using row-wise ConvRot to preserve BF16 quality.")
    precision_group.add_argument('--int4convrot', action='store_const', const='INT4CONVROT', dest='dtype', help="Set output precision to INT4 using row-wise ConvRot for maximum VRAM savings and speed.")
    parser.set_defaults(dtype='BF16')

    precision_main_group.add_argument('--scales-dtype', type=lambda s: s.upper(), choices=['FP32', 'FP16', 'BF16', 'FP16E', 'BF16E'], metavar='TYPE',
                                      help="Precision for quantization scales used in quantized formats. Default: FP32.")
    precision_main_group.add_argument('--mixed-dtype', type=lambda s: s.upper(), choices=['FP32', 'FP16', 'BF16', 'FP16E', 'BF16E'], metavar='TYPE',
                                      help="High precision data type used alongside quantized formats. Default: FP32.")
    precision_main_group.add_argument('--mixed-small', action='store_true',
                                      help="Quantize aggressively for faster speeds at the cost of some quality.")
    precision_main_group.add_argument('--iscales', type=Path, metavar='FILE',
                                      help="Path to a file containing precalculated input scales (only valid with --fp8scaled).")
    precision_main_group.add_argument('--scales-trials', type=int, metavar='NUMBER',
                                      help="Number of candidate scales to try (only valid with --int8convrot).")



    #-- Metadata options group ------------------
    meta_group = parser.add_argument_group('safetensors metadata options')
    meta_group.add_argument('--title'      , metavar='TITLE'  , help="Model title for the safetensors header.")
    meta_group.add_argument('--author'     , metavar='AUTHOR' , help="Author name for the safetensors header.")
    meta_group.add_argument('--description', metavar='TEXT'   , help="Model description for the header.")
    meta_group.add_argument('--license'    , metavar='LICENSE', help="License info for the safetensors header.")
    meta_group.add_argument('--thumbnail'  , metavar='PATH'   , help="Path to the thumbnail image file to include in the header.")

    #-- Advanced options ------------------------
    advanced_group = parser.add_argument_group('advanced options')
    advanced_group.add_argument('--clamp'  , type=str, metavar='LIMIT[:SHARPNESS]',
                                help=("Apply value clipping to weights. Specify as 'limit' or 'limit:sharpness'.\n"
                                      "Example: '7.0:0.8' to set limit 7.0 with a sharpness factor of 0.8."))
    advanced_group.add_argument('--seed', type=int, default=100, metavar='N',
                                help=("This value is used with types involving stochastic rounding, such as --fp16e or --bf16e.\n"
                                      "Setting a fixed seed ensures reproducible results. Default: 100."))

    args = parser.parse_args(parent_args)

    # determine target data type
    target_format = cast(TargetFormat, args.dtype)
    req_quantization, _, dtype_tag, _ = TARGET_FORMAT_INFO[target_format]

    # verify '--iscales' is provided only when '--fp8scaled' is selected
    if args.iscales and args.dtype != 'FP8SCALED':
        parser.error("The --iscales argument can only be used when '--fp8scaled' is selected as the quantization option.")

    # handle "--scales-trials" setting scales_search_trials only when '--int8convrot' was selected
    scales_search_trials = 0
    if args.scales_trials:
        if target_format == 'INT8CONVROT':
            scales_search_trials = args.scales_trials
        else:
            parser.error("The --scales-trials argument can only be used when 'INT8CONVROT' is selected as the quantization option.")

    # check input files and determine model class
    input_files      = validate_and_collect_safetensors(args.input_files)
    input_file_count = len(input_files)
    model = detect_model_architecture(input_files)


    # load the "iscales" file if required by the user
    iscales, iscales_tag = None, ""
    if args.iscales:
        if not args.iscales.is_file():
            parser.error(f"The input scales file does not exist or is not a valid file: {args.iscales}")
        try:
            iscales     = preload_fp8_scales(args.iscales)
            iscales_tag = "_iscales"
        except Exception as e:
            parser.error(f"Failed to load the input scales file '{args.iscales}': {e}")
            iscales, iscales_tag = None, ""


    # determine the high precision data type (default to FP32)
    if req_quantization:
        mixed_small   = bool(args.mixed_small)
        mixed_format  = cast(PrecisionFormat,  args.mixed_dtype  or 'FP32' )
        scales_format = cast(PrecisionFormat,  args.scales_dtype or 'FP32' )
        quality_tag   = "_small" if mixed_small else ""
        mixed_tag     = f"_{mixed_format.lower()}mixed"
        scales_tag    = f"_{scales_format.lower()}qs" if scales_format != 'FP32' else ""
    else:
        mixed_small   = False
        mixed_format  = None
        scales_format = None
        quality_tag   = ""
        mixed_tag     = ""
        scales_tag    = ""
        if args.mixed_small:
            warning("--mixed-small is only supported for quantized models, it will be ignored")
        if args.mixed_dtype is not None:
            warning("--mixed-dtype is only supported for quantized models, it will be ignored")
        if args.scales_dtype is not None:
            warning("--scales-dtype is only supported for quantized models, it will be ignored")

    # determine the clamp parameters (if any)
    clamp_limit, clamp_sharpness = None, None
    if args.clamp:
        clamp_limit, clamp_sharpness = parse_clamp_args(args.clamp)
        if clamp_limit is None or clamp_sharpness is None:
            error("Invalid --clamp argument format. Expected format: 'limit:sharpness'")
            sys.exit(1)
    clamping_tag = make_clamping_tag(clamp_limit, clamp_sharpness)

    # validate model class
    #  - diffusion model -> "z-image"
    #  - text encoder    -> "qwen-3b"
    if model == "z-image":
        tensor_mapper       = ZImageTensorMapper(aggressive_quantization=mixed_small)
        default_name        = 'z_image_turbo.safetensors'
        checkpoint_metadata = ZIMAGE_METADATA

    elif model == "qwen3-4b":
        tensor_mapper       = Qwen3TensorMapper()
        default_name        = 'qwen3-4b.safetensors'
        checkpoint_metadata = QWEN3_4B_METADATA

    else:
        error("Unknown model")
        exit(1)


    # build path to the output safetensors file
    output_path  = Path(args.output or default_name)
    if not output_path.suffix:
        output_path = output_path.with_suffix(".safetensors")

    new_filename = f"{output_path.stem}_{dtype_tag}{mixed_tag}{scales_tag}{quality_tag}{iscales_tag}{clamping_tag}{output_path.suffix}"
    output_path = output_path.with_name(new_filename)


    # create the RNG for stochastic rounding
    stochastic_generator = np.random.default_rng(args.seed)


    # print configuration details
    clamp_limit_str = f"{clamp_limit} (sharpness {clamp_sharpness})" if clamp_limit is not None else "-"
    message(f"Model            : {model.upper()}")
    message(f"Target Data Type : {target_format}")
    if mixed_format is not None:
        message(f"Mixed Data Type  : {mixed_format.upper()}{ ' (SMALL)' if mixed_small else '' }")
    if scales_format is not None:
        message(f"Scales Data Type : {scales_format.upper()}")
    if scales_search_trials:
        message(f"Scales Searches  : {scales_search_trials}")
    message(f"Clamping Value   : {clamp_limit_str}")
    message(f"Stochastic Seed  : {args.seed}")
    message(f"Input            : {input_file_count} safetenstensors {'file' if input_file_count == 1 else 'files'}")
    message(f"Output File      : {output_path.name}")

    # prepare the temporary file for in-memory or disk based on --low-ram argument
    if args.low_ram:
        message("Using disk-based temporary file for low RAM mode.")
        tmp_context = tempfile.TemporaryFile(dir=output_path.parent)
    else:
        message("Using in-memory buffer for temporary data.")
        tmp_context = io.BytesIO()

    # generate the initial info of the safetensors header
    safetensors_header = create_safetensors_header(
        title       = args.title       or checkpoint_metadata["title"],
        author      = args.author      or checkpoint_metadata["author"],
        license     = args.license     or checkpoint_metadata["license"],
        description = args.description or checkpoint_metadata["description"],
        thumbnail_path = args.thumbnail,
        architecture   = checkpoint_metadata["architecture"],
        tags           = checkpoint_metadata["tags"],
        resolution     = checkpoint_metadata["resolution"],
        date           = "*")


    # PROCESS!!
    with tmp_context as tmp_rawtensor_file:

        for input_file in input_files:
            progress_bar = ProgressBar()
            safetensors_header |= convert_safetensors_file(
                                        input_file,
                                        tmp_rawtensor_file,
                                        target_format         = target_format,
                                        scales_format         = scales_format,
                                        high_precision_format = mixed_format,
                                        scales_search_trials  = scales_search_trials,
                                        stochastic_generator  = stochastic_generator,
                                        fp8_preloaded_scales  = iscales,
                                        clamp_limit           = clamp_limit,
                                        clamp_sharpness       = clamp_sharpness,
                                        tensor_mapper         = tensor_mapper,
                                        progress              = progress_bar)

        if args.sort_tensors:
            message("Sorting tensors by size...")
            output_header = sort_safetensors_header(safetensors_header)
        else:
            message("Skipping tensor sorting as requested.")
            output_header = copy.deepcopy(safetensors_header)

        tmp_rawtensor_file.seek(0)
        progress_bar = ProgressBar()
        build_safetensors(output_path,
                            output_header         = output_header,
                            sour_rawtensor_header = safetensors_header,
                            sour_rawtensor_file   = tmp_rawtensor_file,
                            progress              = progress_bar
                            )



if __name__ == "__main__":
    main()
