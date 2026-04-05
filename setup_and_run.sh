#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────
# setup_and_run.sh
# Creates the conda environment and runs the experiment.
# Usage:  bash setup_and_run.sh
# ──────────────────────────────────────────────────────────────
set -e

ENV_NAME="spatial-gan"

# ── 1. Create conda environment ──────────────────────────────
if conda info --envs | grep -qw "$ENV_NAME"; then
    echo "Environment '$ENV_NAME' already exists — skipping creation."
else
    echo "Creating conda environment '$ENV_NAME' ..."
    conda env create -f environment.yml
fi

# ── 2. Activate ──────────────────────────────────────────────
echo "Activating '$ENV_NAME' ..."
eval "$(conda shell.bash hook)"
conda activate "$ENV_NAME"

# ── 3. Verify key packages ──────────────────────────────────
echo ""
echo "Environment check:"
python -c "
import numpy, scipy, sklearn, pandas, torch
print(f'  numpy        {numpy.__version__}')
print(f'  scipy        {scipy.__version__}')
print(f'  scikit-learn {sklearn.__version__}')
print(f'  pandas       {pandas.__version__}')
print(f'  torch        {torch.__version__}')
print(f'  CUDA avail   {torch.cuda.is_available()}')
"

# ── 4. Run ───────────────────────────────────────────────────
echo ""
echo "Running data generation smoke test ..."
python data_generation.py

echo ""
echo "Running full experiment ..."
python experiment.py

echo ""
echo "Done. Results are printed above."
