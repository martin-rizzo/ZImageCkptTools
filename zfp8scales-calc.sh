#!/usr/bin/env bash
# File    : zfp8scales-calc.sh
# Purpose : Wrapper for `zfp8scales-calc.py` that handles the python virtual environment
# Author  : Martin Rizzo | <martinrizzo@gmail.com>
# Date    : Jul 12, 2026
# Repo    : https://github.com/martin-rizzo/ComfyUI-ZImagePowerNodes
# License : MIT
#- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
#                               ZImageCkptTools
#          CLI tools for manipulating and verifying Z-Image checkpoints.
#_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _
RESOLVE_LINK=true                #< `true` to resolve symbolic links to this script
DEDICATED_VENV=false             #< `true` if the script requires a unique virtual env separate from other project scripts
SCRIPT_PATH="${BASH_SOURCE[0]}"  #< full path to this script

# if symbolic links should be resolved, determine the absolute physical path
# using realpath to get the canonical location of the script
if [[ "$RESOLVE_LINK" = true ]]; then 
    SCRIPT_PATH=$(realpath "${BASH_SOURCE[0]}")
fi

SCRIPT_NAME=$(basename "$SCRIPT_PATH" .sh)           #< script name without extension
SCRIPT_DIR=$(cd "$(dirname "$SCRIPT_PATH")" && pwd)  #< script directory
REQUIREMENTS_FILE="${SCRIPT_DIR}/requirements.txt"   #< path to requirements file
DEFAULT_VENV_DIR="${SCRIPT_DIR}/venv"                #< default path for the virtual env directory

# if the script requires its own specific virtual env different from the common one,
# update the paths to use names specific to this script.
if [[ "$DEDICATED_VENV" = true ]]; then
    REQUIREMENTS_FILE="${SCRIPT_DIR}/${SCRIPT_NAME}.requirements"  
    DEFAULT_VENV_DIR="${SCRIPT_DIR}/venv-${SCRIPT_NAME}"
fi

# VENV_DIR: specifies the directory for python virtual environment; default is `SCRIPT_DIR/venv`
# PYTHON  : specifies the path to the Python interpreter; default is `python3`
[[ "$VENV_DIR" ]] || VENV_DIR="$DEFAULT_VENV_DIR"
[[ "$PYTHON"   ]] || PYTHON=python3
PYTHON_SCRIPT="${SCRIPT_DIR}/${SCRIPT_NAME}.py"          #< path to python script to run

# List of options that do not trigger any action by themselves
NON_ESSENTIAL_OPTIONS=( "-c" "--color" "--color-always" )

# ANSI escape codes for colored terminal output
RED='\e[1;31m'; CYAN='\e[1;36m'; YELLOW='\e[93m'; RESET='\e[0m'


#============================ HELPER FUNCTIONS =============================#

# Display a warning message
warning() { echo -e "\n${CYAN}[${YELLOW}WARNING${CYAN}]${RESET} $1" >&2; }

# Display an error message
error() { echo -e "\n${CYAN}[${RED}ERROR${CYAN}]${RESET} $1" >&2; }

# Displays a fatal error message and exits the script
fatal_error() {
    error "$1"; shift
    while [[ $# -gt 0 ]]; do
        echo -e " ${CYAN}\xF0\x9F\x9B\x88 $1${RESET}" >&2
        shift
    done
    echo; exit 1
}

# Check if a given option is non-essential
# (non-essential options do not trigger any action by themselves)
is_non_essential_option() {
    local option=$1
    [[ -z "$option" ]] && return 0
    for non_essential_option in "${NON_ESSENTIAL_OPTIONS[@]}"; do
        [[ "$option" == "$non_essential_option" ]] && return 0
    done
    return 1
}

#=========================== VIRTUAL ENVIRONMENT ===========================#

# Create and activate the python virtual environment
create_venv() {
    if [[ -d "$VENV_DIR" ]]; then
        echo "Virtual environment already exists."
        return
    fi
    echo "Creating virtual environment..."
    if ! python3 -m venv "$VENV_DIR"; then
        fatal_error "Virtual environment creation failed." \
                    "Please check if python3 and venv are installed on your system."
    fi
    echo "Virtual environment created."
}

# Remove the python virtual environment
remove_venv() {
    if [[ ! -d "$VENV_DIR" ]]; then
        fatal_error "No 'venv' directory found." \
                    "You must create a virtual environment before removing it." \
                    "Use the '--create-venv' option to create a new one."
    fi
    rm -rf "$VENV_DIR"
    echo "Virtual environment removed."
}

# Activate the python virtual environment
activate_venv() {
    if [[ ! -f "$VENV_DIR/bin/activate" ]]; then
        fatal_error "The virtual environment does not exist." \
                    "you can use --create-venv to create it"
    fi
    # shellcheck disable=SC1091
    if ! source "$VENV_DIR/bin/activate"; then
        fatal_error "Error when activating virtual environment, it might be corrupted." \
                    "You can use --recreate-venv to recreate the virtual environment."
    fi
}

# Install dependencies from requirements.txt file if it exists
install_dependencies() {
    local requirements_file=$1
    if [[ ! -f "$requirements_file" ]]; then
        fatal_error "No '$requirements_file' file found." \
                    "Please check the project instalation instructions."
    fi
    if ! pip install --upgrade pip; then
        # failed to upgrade pip isn´t a fatal error, just a warning
        warning "Error when upgrading pip."
    fi
    if ! pip install -r "$requirements_file"; then
        fatal_error "Error when installing dependencies." \
                    "'pip' failed to install some packages, that might be due to network issues or incompatible packages."
    fi
    echo "Dependencies installed successfully."
}

#===========================================================================#
#////////////////////////////////// MAIN ///////////////////////////////////#
#===========================================================================#

# verify if any extra options are passed as arguments
CREATE_VENV=false
REMOVE_VENV=false
SHOW_HELP=false

if [[ $# -le 1 ]] && is_non_essential_option "$1"; then
    # if no arguments are passed, the help message will be displayed
    SHOW_HELP=true
else
    # loop through the arguments and set the corresponding
    # variables to true if they match the options
    for arg in "$@"; do
        case $arg in
            -h | --help)
                SHOW_HELP=true
                ;;
            --create-venv)
                CREATE_VENV=true
                ;;
            --remove-venv)
                REMOVE_VENV=true
                ;;
            --recreate-venv)
                REMOVE_VENV=true
                CREATE_VENV=true
                ;;
        esac
    done
fi

# handle the help option
if [[ "$SHOW_HELP" == true ]]; then
    python3 "$PYTHON_SCRIPT" --help
    echo
    echo -e "${RED}wrapper options:${RESET}"
    echo -e "  ${CYAN}--create-venv${RESET}      Create the python virtual environment"
    echo -e "  ${CYAN}--remove-venv${RESET}      Remove the python virtual environment"
    echo -e "  ${CYAN}--recreate-venv${RESET}    Remove and recreate the python virtual environment"
    echo
    exit 0
fi

# if either create or remove options are requested, execute and exit
if [[ "$CREATE_VENV" == true || "$REMOVE_VENV" == true ]]
then
    # handle the option for removing the venv
    if [[ "$REMOVE_VENV" == true ]]; then
        remove_venv
    fi
    # handle the option for creating the venv
    if [[ "$CREATE_VENV" == true ]]; then
        create_venv
        activate_venv
        install_dependencies "$REQUIREMENTS_FILE"
    fi
    exit 0
fi

# if no extra options are passed, just run the script normally
if [[ ! -f "$PYTHON_SCRIPT" ]]; then
    python_script_name=$(basename "$PYTHON_SCRIPT")
    fatal_error "Python script not found." \
                "Please ensure that the Python script '${python_script_name}' exists in the same directory as this bash wrapper."
fi
activate_venv
"$PYTHON" "$PYTHON_SCRIPT" "$@"
