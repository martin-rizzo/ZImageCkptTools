# ZModelTools

A set of command-line tools written in Python for managing, modifying, and verifying Z-Image and Z-Image Turbo checkpoints. These tools are designed primarily to make the checkpoints compatible for use in ComfyUI. This is currently a personal experimental project, so it may fail and is not fully documented. The tools are designed and tested on Linux (Fedora), although the Python scripts should run on any platform.

## Tools

- `zcomfymake`: Packages separate Diffusers files into a unified checkpoint compatible with ComfyUI and WebUI.
- `zcheck`: Verifies a checkpoint to ensure that its key tensors have not suffered excessive quantization.

### Prerequisites
- Python 3.11 or higher
- pip (Python package installer)

### Installation
- Once the code is downloaded locally, the scripts can be run directly using the bash wrappers, which automatically set up the virtual environment using the `--create-venv` parameter. (See `--help` for more information).
- On platforms other than Linux, run the Python scripts directly.
