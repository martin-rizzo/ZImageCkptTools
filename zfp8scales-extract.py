"""
File    : zfp8scales-extract.py
Purpose : Extract ".scale_input" values from a float8 scaled quantized safetensors checkpoint
Author  : Martin Rizzo | <martinrizzo@gmail.com>
Date    : Jul 15, 2026
Repo    : https://github.com/martin-rizzo/ZImageCkptTools
License : MIT
- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                               ZImageCkptTools
          CLI tools for manipulating and verifying Z-Image checkpoints.
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _
"""
import re
import sys
import argparse
from typing   import Any, TextIO
from pathlib  import Path
SCRIPT_NAME = Path(__file__).stem
if __name__ == '__main__' and ("-h" not in sys.argv and "--help" not in sys.argv):

    # Modules that are not available in the standard library must be imported here
    from safetensors import safe_open
    import numpy as np
    import ml_dtypes


# ANSI escape codes for colored terminal output
RED      = '\033[91m'
DKRED    = '\033[31m'
YELLOW   = '\033[93m'
DKYELLOW = '\033[33m'
GREEN    = '\033[92m'
CYAN     = '\033[96m'
DKGRAY   = '\033[90m'
RESET    = '\033[0m'


#================================ MESSAGES =================================#

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

def _get_unique_path(path: Path) -> Path:
    """Generate a unique filename by appending an incrementing suffix."""
    if not path.exists():
        return path

    counter       = 0
    original_stem = path.stem
    while True:
        counter += 1
        new_path = path.with_stem(f"{original_stem}_{counter}")
        if not new_path.exists():
            return new_path


def _sort_tensors(tensors: dict[str, Any]) -> tuple[list[tuple[str, Any]], list[tuple[str, Any]]]:
    """
    Splits tensors into metadata and actual tensor items, sorting tensor names by natural numerical order.

    Args:
        tensors: A dictionary mapping tensor names or metadata to their values.
    Returns:
        A tuple containing two lists of (key, value) pairs:
          1. metadata_items: Items starting with '__' (unversioned/unsorted).
          2. tensor_items  : Regular tensor items, sorted naturally by numerical index.
    """
    def natural_sort_key(item_tuple: tuple[str, Any]) -> list[Any]:
        key_string = item_tuple[0]
        return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', key_string)]

    metadata_items: list[tuple[str, Any]] = []
    layer_items   : list[tuple[str, Any]] = []

    # separate items into metadata and tensor categories
    for layer_name, value in tensors.items():
        if layer_name.startswith('__'):
            metadata_items.append((layer_name, value))
        else:
            layer_items.append((layer_name, value))

    # sort tensor items using natural numerical key
    layer_items.sort(key=natural_sort_key)
    return metadata_items, layer_items


#=========================== FP8 SCALE UTILITIES ===========================#

def _extract_fp8_scales(file_path: Path) -> dict[str, float]:
    """
    Search for tensors ending with '.scale_input' in a safetensors file.

    If the tensor contains a single value, it is stored in a dictionary.
    If it contains more than one value, a warning is issued and the tensor is skipped.

    Args:
        file_path : The path to the .safetensors file to be processed.
    Returns:
        A dictionary where keys are the tensor names and values are the extracted scalars.
    """
    found_scales = {}
    with safe_open(file_path, framework="np", device="cpu") as safetensors_file:

        for tensor_name in safetensors_file.keys():

            if not tensor_name.endswith('.scale_input'):
                continue

            # calculate the total number of elements without loading the whole tensor
            tensor_slice = safetensors_file.get_slice(tensor_name)
            num_elements = np.prod(tensor_slice.get_shape())

            # check if the tensor has only one element
            # scale tensor is a special case, so we need to check for it
            if num_elements != 1:
                warning(f"The tensor '{tensor_name}' has {num_elements} elements. "
                        f"Expected only 1 value. Skipping...")
                continue

            # load the scale tensor and get its value
            scale_tensor = safetensors_file.get_tensor(tensor_name)
            found_scales[tensor_name] = float(scale_tensor.item())

    return found_scales


def _write_fp8_scales(input_scales: dict[str, Any],
                      *,
                      file        : TextIO,
                      header_text : str = "",
                      ) -> None:
    """
    Write the calculated FP8 input scales to a specified output stream.

    Iterates through the input_scales dictionary and prints the scale factor
    for each layer in a formatted table.

    Args:
        input_scales : A dictionary mapping layer names to their computed
                       FP8 quantization scale factors, including metadata
                       keys (e.g., '__dtype__', '__count__').
        file         : A file-like object (e.g., sys.stdout or an open file)
                       where the output will be written.
        header_text  : An optional string to print as a title or description
                       at the beginning of the file.

    Returns:
        None
    """
    if header_text:
        print(header_text, file=file)

    print(file=file)
    metadata_items, layer_items = _sort_tensors(input_scales)

    for name, value in metadata_items:
        print(f"{name}: {value}", file=file)

    max_length = max([len(layer_name) for layer_name, _ in layer_items], default=1)
    for layer_name, input_scale in layer_items:
        print(f"{layer_name:<{max_length}} : {input_scale}", file=file)



#===========================================================================#
#////////////////////////////////// MAIN ///////////////////////////////////#
#===========================================================================#

def main(parent_args  : list[str] | None = None,
         parent_script: str | None       = None
         ) -> int:
    """
    Main entry point for the CLI tool.
    Args:
        parent_args   : List of arguments to parse or `None` for reading from command line.
        parent_script : The name of the calling script if any.
    """
    parser = argparse.ArgumentParser(
        prog        = f"{parent_script} {SCRIPT_NAME}" if parent_script else SCRIPT_NAME,
        description = (
            "Extracts '.scale_input' tensors from a float8 quantized safetensors checkpoint.\n"
            "This tool identifies all keys ending in '.scale_input', extracts their values, \n"
            "and saves them as a text-based format or prints them directly to stdout."
        ),
        formatter_class = argparse.RawTextHelpFormatter,
    )
    parser.add_argument('input_path'    , type=Path, help="Path to the source .safetensors file containing float8 scale parameters.")
    parser.add_argument('-o', '--output', type=str,
                        help=("Output file path for the extracted scales. If not provided, it defaults to\n"
                              "'<input_filename>.input_scales.txt'. Use '-' to print results to the console."))
    args = parser.parse_args(parent_args)

    # validate input file existence
    input_path = Path(args.input_path)
    if not input_path.is_file():
        error(f"The input safetensors file '{input_path}' does not exist.")
        return 1


    fp8_scales = _extract_fp8_scales(input_path)
    if not fp8_scales:
        error(f"The input safetensors file '{input_path}' does not contain fp8 scales.")
        return 1

    # if the user used "-o -", redirects output directly to console stdout
    if args.output == '-':
        _write_fp8_scales(fp8_scales, file=sys.stdout)
        return 0

    # build the output file name if not specified
    output_path = args.output
    if not output_path:
        output_path = input_path.with_suffix('.fp8scales.txt')

    # write the output file preventing overwriting
    output_path = _get_unique_path(output_path)
    with open(output_path, "w", encoding="utf-8") as f:
        _write_fp8_scales(fp8_scales, file=f)
        print(f"FP8 scales were written to {output_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
