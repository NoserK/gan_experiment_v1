# Bivariate Spatial Simulation & GAN Estimation

## Quick Start

```bash
# 1. Create the conda environment
conda env create -f environment.yml

# 2. Activate it
conda activate spatial-gan

# 3. Generate data (writes CSVs to non_stationary/ and non_gaussian/)
python data_generation.py

# 4. Run the estimation experiment
python experiment.py
```

Or simply run everything at once:

```bash
bash setup_and_run.sh
```

## File Overview

| File | Purpose |
|---|---|
| `environment.yml` | Conda environment specification |
| `setup_and_run.sh` | One-command setup + run script |
| `data_generation.py` | Python port of the R data generation program |
| `experiment.py` | cGAN / KNN / Random Forest estimation with metrics |
| `data_generation_scheme.tex` | LaTeX source for the mathematical writeup |
| `data_generation_scheme.pdf` | Compiled PDF of the data generation scheme |

## Environment Details

- **Name:** `spatial-gan`
- **Python:** 3.11
- **Key packages:** numpy, pandas, scipy, scikit-learn, pytorch, matplotlib

## GPU Support

The cGAN automatically uses CUDA if available. For CPU-only machines, everything
runs identically — just slower. To install the CUDA-enabled PyTorch variant, replace
the pip torch line in `environment.yml` with:

```yaml
  - pip:
      - torch>=2.0 --index-url https://download.pytorch.org/whl/cu121
```

## Configuration

In `experiment.py`, line ~170:

```python
num_sim = 10   # set to 50 for the full Monte-Carlo study
```

Increase to 50 to match the original R simulation design.

## Removing the Environment

```bash
conda deactivate
conda env remove -n spatial-gan
```
