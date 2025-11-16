# Copyright 2025 Geoffrey Taghon

echo ""
echo "================================================================="
echo "Openfold3-MLX Installation script - v0.1.0"
echo "================================================================="
echo ""

# Install Openfold3-MLX
python3 -m venv .venv
source ./.venv/bin/activate && pip install -e .

echo ""
echo "================================================================="
echo "Openfold3-MLX Installation script completed."
echo "Test protein inference with: ./predict.sh <sequence> completed."
echo "================================================================="
echo ""