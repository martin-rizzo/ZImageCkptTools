#!/usr/bin/env python3
"""
File    : zinputscale-calc.py
Purpose : Calculate the optimal ".scale_input" factor for ComfyUI-compatible
          float8 scaled quantization, extracting information from imatrix files
Author  : Martin Rizzo | <martinrizzo@gmail.com>
Date    : Jul 12, 2026
Repo    : https://github.com/martin-rizzo/ZImageCkptTools
License : MIT
- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                               ZImageCkptTools
          CLI tools for manipulating and verifying Z-Image checkpoints.
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _


 Content of 'sum_squared_activations' (mapped from `sums` in the imatrix):
 -------------------------------------------------------------------------
 This field stores a vector of F32 values representing the cumulative sum of
 squared activations for a specific layer.

 Mechanism:
   1. During 'llama-imatrix' calibration, input data is processed in chunks.
   2. The magnitude of each layer's activation is squared to capture the
      absolute impact.
   3. These values are accumulated into `sum_squared_activations` across all
      passes. The total number of samples is recorded in `total_samples`.

 Estimating Maximum Range (amax) from RMS
 ----------------------------------------
 Since the imatrix collapses individual peak values into a sum of squares, we
 apply a statistical correction factor to estimate the distribution tail (amax).
 Assuming a Gaussian distribution centered at 0, the RMS serves as the standard
 deviation (sigma).
 ```python

    # calculate mean of squares
    mean_of_squares = imatrix_entry["sum_squared_activations"] / imatrix_entry["total_samples"]
    # calculate "Root Mean Square (RMS)"
    rms_values = np.sqrt(mean_of_squares)
    # scale by a statistical factor (typically 3.0 to 4.5) to estimate the
    # absolute maximum (amax) without severe clipping.
    estimated_amax = rms_values * sigma_factor

 ```
"""
import os
import sys
import argparse
from typing import Any
from pathlib import Path
if __name__ == '__main__' and ("-h" not in sys.argv and "--help" not in sys.argv):
    # Modules that are not available in the standard library must be imported here
    import numpy as np
    import ml_dtypes

    # Dictionary of maximum values for different FP8 formats
    FP8_MAX_VALUES = {
        "fp8_e4m3": float(ml_dtypes.finfo("float8_e4m3fn").max),
        "e4m3"    : float(ml_dtypes.finfo("float8_e4m3fn").max),
        "fp8_e5m2": float(ml_dtypes.finfo("float8_e5m2").max),
        "e5m2"    : float(ml_dtypes.finfo("float8_e5m2").max),
    }

    # Conversion table from GGUF tensor names to Safetensors tensor names.
    # This table has been tested with Qwen3-4b but should work for other models in the family
    QWEN3_GGUF_TO_SAFETENSORS = {
        ".blk."        : ".model.layers."    ,
        ".ffn_down."   : ".mlp.down_proj."   ,
        ".ffn_up."     : ".mlp.up_proj."     ,
        ".ffn_gate."   : ".mlp.gate_proj."   ,
        ".attn_output.": ".self_attn.o_proj.",
        ".attn_k."     : ".self_attn.k_proj.",
        ".attn_q."     : ".self_attn.q_proj.",
        ".attn_v."     : ".self_attn.v_proj.",
    }


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

def _fix_tensor_names(imatrix: dict, *, table: dict) -> dict:
    """
    Renames GGUF tensor keys to Safetensors format using a provided conversion table.

    Args:
        imatrix: Input dictionary where keys are GGUF tensor names
                 and values are the corresponding tensor data.
        table  : Dictionary containing a mapping of GGUF subkey to their
                 equivalent Safetensors subkeys used for renaming.
    Returns:
        Dictionary with tensor keys renamed to Safetensors format, with the
        original tensor data preserved.
    """
    result = {}
    for tensor_name, data in imatrix.items():
        # apply substring replacements by iterating over the conversion table
        # (a leading dot is temporarily added to ensure replacements work correctly)
        new_name = f".{tensor_name}"
        for gguf_key, safetensor_key in table.items():
            if gguf_key in new_name:
                new_name = new_name.replace(gguf_key, safetensor_key)
        result[new_name.lstrip('.')] = data

    return result


#================================= IMATRIX =================================#

def _load_legacy_imatrix(path: Path) -> dict[str, dict]:
    """
    Load a legacy imatrix binary file into a dictionary.

    The code used by llama.cpp to perform this loading can be found here:
    - https://github.com/ggml-org/llama.cpp/blob/b9980/common/imatrix-loader.cpp#L10

    The expected binary structure is:
    - 4 bytes: Total number of tensors in the file (int32)
    Per-tensor structure:
        - 4 bytes: Length of the tensor name (int32)
        - N bytes: UTF-8 encoded tensor name (N bytes)
        - 4 bytes: Number of samples recorded for the tensor (int32)
        - 4 bytes: Dimension of the activation data for the tensor (int32)
        - M bytes: Raw float32 values representing the sum of squared activations (M = dimension * 4)

    Args:
        path: Path pointing to the input legacy imatrix binary file.

    Returns:
        Dictionary mapping tensor names to their associated data:
        {
            "<tensor_name>": {
                "sum_squared_activations": np.ndarray,  # 1D array of float32 quadratic activation sum values
                "total_samples"          : int          # Total forward pass count for the tensor
            },
        }
    """
    # limits to detect corrupted or unknown file formats
    MAX_ALLOWED_TENSOR_COUNT = 4096  #< Reasonable maximum number of tensors for a valid legacy imatrix file
    MAX_TENSOR_NAME_LENGTH   =  512  #< Reasonable maximum length (in bytes) for a tensor name.

    file_size = path.stat().st_size
    max_float32_entries = file_size // 4
    result: dict[str, dict] = {}
    with open(path, "rb") as imatrixfile:
        # read global tensor count header
        bytes = imatrixfile.read(4)
        if len(bytes) != 4:
            raise ValueError("Missing global tensor count header. Corrupted file?")
        n_tensors = int(np.frombuffer(bytes, count=1, dtype=np.int32)[0])

        # validate tensor count is in a logical range to detect unknown or corrupt file formats
        if n_tensors <= 0 or n_tensors > MAX_ALLOWED_TENSOR_COUNT:
            raise ValueError(f"Suspicious tensor count: {n_tensors}. Invalid file format")

        for _ in range(n_tensors):
            # read length of the current tensor name
            bytes = imatrixfile.read(4)
            if len(bytes) < 4:
                break
            name_len = int(np.frombuffer(bytes, count=1, dtype=np.int32)[0])

            # validate name length is in a logical range to detect unknown or corrupt file formats
            if name_len <= 0 or name_len > MAX_TENSOR_NAME_LENGTH:
                raise ValueError(f"Suspicious tensor name length: {name_len}. Invalid file format")

            # read tensor name bytes
            bytes = imatrixfile.read(name_len)
            if len(bytes) < name_len:
                raise ValueError(f"Incomplete tensor name. Corrupted file?")
            tensor_name = bytes.decode("utf-8", errors="ignore")

            # read number of forward passes (samples) for the current layer
            bytes = imatrixfile.read(4)
            if len(bytes) < 4:
                raise ValueError(f"Missing activation count for tensor {tensor_name}. Corrupted file?")
            total_samples = int(np.frombuffer(bytes, count=1, dtype=np.int32)[0])

            # read the activation dim for the current tensor
            bytes = imatrixfile.read(4)
            if len(bytes) < 4:
                raise ValueError(f"Missing entry count for tensor {tensor_name}. Corrupted file?")
            n_entries = int(np.frombuffer(bytes, count=1, dtype=np.int32)[0])

            # validate entry count does not exceed the maximum possible entries for the file size
            if n_entries > max_float32_entries:
                raise ValueError(f"Tensor {tensor_name} has {n_entries} entries, more than the file can contain.")

            # read the activation data for the current tensor
            bytes = imatrixfile.read(n_entries * 4)
            if len(bytes) < n_entries * 4:
                raise ValueError(f"Incomplete data for tensor {tensor_name}. Corrupted file?")
            sum_squared_activations = np.frombuffer(bytes, count=n_entries, dtype=np.float32)

            # add the tensor data to the result dictionary
            result[tensor_name] = {
                "sum_squared_activations" : sum_squared_activations,
                "total_samples"           : total_samples,
            }

    return result



def _calculate_fp8_input_scale(sum_squared_activations: np.ndarray | list[float] | float,
                               total_samples          : int,
                               sigma           : float = 3.7,
                               dtype                  : str   = 'fp8_e4m3'
                               ) -> float:
    """
    Calculate the input scale factor for FP8 (E4M3FN format) quantization
    using an estimation of the activation matrix root mean square (RMS).

    This function estimates the maximum absolute value (amax) of the activation
    distribution tail to select the appropriate scale that minimizes clipping
    of outlier activation values during quantization.

    Args:
        sum_squared_activations : Aggregated sum of squared activation values, typically computed per channel across all input samples.
        total_samples           : Total number of samples used to compute the aggregated sum of squared activations.
        sigma_factor            : Scaling factor applied to the representative global RMS to estimate the analytical amax
                                  (cutoff point for the distribution tail). Defaults to 3.7.
        dtype                   : Desired FP8 format, can be 'fp8_e4m3' or 'fp8_e5m2'. Defaults to 'fp8_e4m3'.

    Returns:
        The final input scale factor to apply for FP8 quantization.
    """
    sum_squares     = np.array(sum_squared_activations, dtype=np.float32)
    mean_of_squares = sum_squares / total_samples
    rms_values      = np.sqrt(mean_of_squares)

    # select the maximum RMS value across all channels
    global_rms = float(np.max(rms_values))

    # estimate the analytical amax (cutoff point for the distribution tail)
    estimated_amax = global_rms * sigma

    # get the maximum value for the given FP8 format
    fp8_max = FP8_MAX_VALUES.get(dtype.lower().strip())
    if not fp8_max:
        raise ValueError(f"Unsupported FP8 format: '{dtype}'. Use 'fp8_e4m3' or 'fp8_e5m2'.")

    # compute the final multiplicative input scale
    input_scale = estimated_amax / fp8_max
    return input_scale



#===========================================================================#
#////////////////////////////////// MAIN ///////////////////////////////////#
#===========================================================================#

def main(args=None, parent_script=None):
    """
    Main entry point for the imatrix2scale CLI tool.
    Args:
        args          (optional): List of arguments to parse. Default is None.
        parent_script (optional): The name of the calling script if any.
    """
    prog = "imatrix2scale"
    if parent_script:
        prog = f"{parent_script} {os.path.splitext(os.path.basename(__file__))[0]}"

    parser = argparse.ArgumentParser(
        prog        = prog,
        description = (
            "Extract activation profiles from llama.cpp imatrix files and convert them\n"
            "into static '.scale_input' tensors compatible with FP8-scaled network topologies.\n\n"
            "This utility acts as a bridge between GGUF profiling data and FP8 execution backends."
        ),
        formatter_class = argparse.RawTextHelpFormatter,
    )

    #-- Input / Output Arguments ----------------
    parser.add_argument('input_file', metavar='INPUT_IMATRIX', help="Path to the source 'imatrix.dat' calibration file.")
    parser.add_argument('-o', '--output', help="Output file path for the generated scales (Default: same name as input with '.scales' extension)")

    #-- Mathematical & Calibration Options ------
    calib_group = parser.add_argument_group('mathematical & calibration options')
    calib_group.add_argument('--sigma', type=float, default=3.3, metavar='FACTOR',
                             help=("Statistical multiplier (Sigma) to estimate the activation absmax from RMS values.\n"
                                   "3.3 roughly maps to the 99.9th percentile to isolate rare outliers. Default: 3.3."))

    parsed_args = parser.parse_args(args=args)

    # Validate input file existence
    input_path = Path(parsed_args.input_file)
    if not input_path.is_file():
        error(f"The input calibration file '{input_path}' does not exist.")
        sys.exit(1)

    # Determine default names and enforce extension
    default_name = "scales_fp8.safetensors"
    output_path = Path(parsed_args.output or default_name)
    if not output_path.suffix:
        output_path = output_path.with_suffix(".safetensors")

    # Log operational configuration details
    message(f"Tool Name      : {prog.upper()}")
    message(f"Source IMatrix : {input_path.name}")
    message(f"Sigma Factor   : {parsed_args.sigma}")
    message(f"Output File    : {output_path.name}")
    message("----------------------------------------------------------------------")

    # PROCESS!!
    #try:

    imatrix = _load_legacy_imatrix(input_path)
    if not imatrix:
        error("No valid tensor layers could be parsed from the provided imatrix file.")
        sys.exit(1)

    imatrix = _fix_tensor_names(imatrix, table=QWEN3_GGUF_TO_SAFETENSORS)

    print()
    for layer_name, layer_data in imatrix.items():
        sum_squared_activations = layer_data["sum_squared_activations"]
        total_samples           = layer_data["total_samples"]
        input_scale             = _calculate_fp8_input_scale(sum_squared_activations, total_samples, sigma=parsed_args.sigma)
        print(f"{layer_name:<40}: { 2 * input_scale :.8f}")

    # output_np_path = output_path.with_suffix(".npy")
    # tensor_payload = {
    #     f"{layer}.scale_input": np.array([value], dtype=np.float32)
    #     for layer, value in scales_dict.items()
    # }
    # np.save(str(output_np_path), tensor_payload)

    # except Exception as e:
    #     error(f"An unexpected error occurred during processing pipeline: {str(e)}")
    #     sys.exit(1)




if __name__ == "__main__":
    main()