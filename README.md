# ZImageCkptTools

A set of command-line tools written in Python for managing, quantizing, and verifying Z-Image and Z-Image Turbo checkpoints.

These tools were initially created to convert the original checkpoints provided by the model creators on Hugging Face into formats compatible with ComfyUI. This is currently a personal experimental project, so it may encounter issues and is not fully documented. The tools are designed and tested on Linux (Fedora), though the Python scripts should run on any platform.

## Tools

- `z2comfy.py`: Converts Z-Image checkpoint files into various formats compatible with ComfyUI.
- `zfp8scales-calc.py`: Calculates '.scale_input' values compatible with ComfyUI from activation profiles in llama.cpp imatrix files.
- `zfp8scales-extract.py`: Extracts '.scale_input' tensors from a float8 quantized safetensors checkpoint.

### Prerequisites
- Python 3.11 or higher
- pip (Python package installer)

### Installation
- After downloading the code locally, the scripts can be run directly using the bash wrappers, which automatically set up the virtual environment with the `--create-venv` parameter. (See `--help` for more information).
- On platforms other than Linux, you can run the Python scripts directly if the dependencies are installed, and it should work normally.
