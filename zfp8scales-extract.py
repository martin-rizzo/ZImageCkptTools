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
import argparse
from pathlib import Path
SCRIPT_NAME = Path(__file__).stem

#===========================================================================#
#////////////////////////////////// MAIN ///////////////////////////////////#
#===========================================================================#

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

    if args.output:
        output_path = args.output
    else:
        output_path = args.input_path.with_suffix('.input_scales.txt')

    print(f"Processing input file: {args.input_path}")
    print(f"Target output: {output_path}")



if __name__ == "__main__":
    main()