#!/bin/bash

# Installer for Openfold3-MLX
python3 -m venv .venv
source ./venv/bin/activate

# Install Openfold3-MLX
pip install -e .

# Install dependencies
pip install git+https://github.com/TimoLassmann/kalign.git

echo ""
echo "================================================================="
echo "Openfold3-MLX Installation script completed."
echo "Test protein inference with: ./predict.sh <sequence> completed."
echo ""