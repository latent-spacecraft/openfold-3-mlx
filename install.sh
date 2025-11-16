#!/bin/bash

# Install Openfold3-MLX
source ./venv/bin/activate && pip install -e .

# Install dependencies
echo ""
echo "Installing kalign for inference..."
source ./venv/bin/activate && pip install git+https://github.com/TimoLassmann/kalign.git

echo ""
echo "================================================================="
echo "Openfold3-MLX Installation script completed."
echo "Test protein inference with: ./predict.sh <sequence> completed."
echo ""