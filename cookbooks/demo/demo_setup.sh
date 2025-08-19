#!/bin/bash
 
echo "Setting up Python virtual environment (.venv)..."

# Resolve paths relative to this script's directory
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
requirements_file="${script_dir}/requirements.txt"
repo_root="$(cd "${script_dir}/.." && pwd)"

# Try to activate an existing venv (.venv/bin/activate on Unix, .venv/Scripts/activate on Windows Git Bash)
try_activate() {
  if [ -f ".venv/bin/activate" ]; then
    . ".venv/bin/activate" >/dev/null 2>&1 || return 1
  elif [ -f ".venv/Scripts/activate" ]; then
    . ".venv/Scripts/activate" >/dev/null 2>&1 || return 1
  else
    return 1
  fi

  if [ -z "${VIRTUAL_ENV:-}" ]; then
    return 1
  fi

  return 0
}

# Locate a Python 3 interpreter
find_python() {
  if command -v python3 >/dev/null 2>&1; then
    echo "python3"
    return 0
  fi
  if command -v python >/dev/null 2>&1; then
    echo "python"
    return 0
  fi
  if command -v py >/dev/null 2>&1; then
    echo "py -3"
    return 0
  fi
  return 1
}

# Create and activate a new venv
create_and_activate() {
  local pycmd
  pycmd="$(find_python)" || {
    echo "Error: Python 3 not found. Please install Python 3 and re-run this script." >&2
    return 1
  }

  echo "Creating new virtual environment with: ${pycmd}"
  if [ "${pycmd}" = "py -3" ]; then
    py -3 -m venv .venv || return 1
  else
    "${pycmd}" -m venv .venv || return 1
  fi

  try_activate || {
    echo "Error: Failed to activate the new virtual environment." >&2
    return 1
  }

  echo "Virtual environment created and activated at: ${VIRTUAL_ENV}"
}

# Main logic
if [ -d ".venv" ]; then
  if try_activate; then
    echo "Activated existing virtual environment at: ${VIRTUAL_ENV}"
  else
    echo ".venv exists but activation failed. Recreating the virtual environment..."
    rm -rf .venv
    create_and_activate || exit 1
  fi
else
  create_and_activate || exit 1
fi

# Install Python dependencies
if [ -f "${requirements_file}" ]; then
  echo "Installing dependencies from: ${requirements_file}"
  python -m pip install --upgrade pip || {
    echo "Warning: Failed to upgrade pip; proceeding with current version." >&2
  }
  python -m pip install -r "${requirements_file}" || {
    echo "Error: Failed to install Python dependencies from requirements.txt" >&2
    exit 1
  }
else
  echo "Warning: requirements.txt not found at ${requirements_file}. Skipping dependency installation."
fi

# Ensure Piper voice model exists in a gitignored assets folder
piper_assets_dir="${repo_root}/assets/piper_voices"
voice_name="en_US-amy-low"
voice_model_path="${piper_assets_dir}/${voice_name}.onnx"

mkdir -p "${piper_assets_dir}" || {
  echo "Error: Failed to create assets directory: ${piper_assets_dir}" >&2
  exit 1
}

if [ ! -f "${voice_model_path}" ]; then
  echo "Piper voice '${voice_name}' not found. Downloading to ${piper_assets_dir} ..."
  python -m piper.download_voices --output-dir "${piper_assets_dir}" "${voice_name}" || {
    echo "Error: Failed to download Piper voice '${voice_name}'." >&2
    exit 1
  }
else
  echo "Piper voice already present at: ${voice_model_path}. Skipping download."
fi

# Start JupyterLab from './demo' under repo root if it exists, else from repo root
notebook_dir="${repo_root}/demo"
if [ -d "${notebook_dir}" ]; then
  cd "${notebook_dir}" || {
    echo "Error: Failed to change directory to: ${notebook_dir}" >&2
    exit 1
  }
else
  echo "Warning: ${notebook_dir} not found; starting JupyterLab from repo root."
  cd "${repo_root}" || {
    echo "Error: Failed to change directory to repo root: ${repo_root}" >&2
    exit 1
  }
fi
echo "Starting JupyterLab in ${PWD} ..."
exec python -m jupyterlab
