#!/bin/bash

# Script to fix CUDA llama-cpp-python installation
# This script sets up CUDA repository, installs CUDA toolkit, activates the virtual environment and reinstalls llama-cpp-python with CUDA support

echo "Setting up CUDA repository and installing CUDA toolkit..."
wget https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt-get update
sudo apt-get -y install cuda-toolkit-12-6

if [ $? -ne 0 ]; then
    echo "Error: Failed to install CUDA toolkit"
    exit 1
fi

echo "CUDA toolkit installation complete!"

echo "Activating virtual environment..."
source .venv/bin/activate

if [ $? -ne 0 ]; then
    echo "Error: Failed to activate virtual environment"
    exit 1
fi

echo "Setting CUDA environment variables..."
export CMAKE_ARGS="-DGGML_CUDA=on -DCUDA_PATH=/usr/local/cuda-12.6 -DCUDAToolkit_ROOT=/usr/local/cuda-12.6 -DCUDAToolkit_INCLUDE_DIR=/usr/local/cuda-12/include -DCUDAToolkit_LIBRARY_DIR=/usr/local/cuda-12.6/lib64"
export CUDACXX=/usr/local/cuda-12.6/bin/nvcc

echo "Environment variables set:"
echo "CMAKE_ARGS: $CMAKE_ARGS"
echo "CUDACXX: $CUDACXX"

echo "Installing llama-cpp-python with CUDA support..."
pip install llama-cpp-python --upgrade --force-reinstall --no-cache-dir

if [ $? -eq 0 ]; then
    echo "Successfully installed llama-cpp-python with CUDA support!"
else
    echo "Error: Failed to install llama-cpp-python"
    exit 1
fi

echo "Installation complete!"
