# aerobo
Autoencoder-enhanced Joint Input-Output Dimensionality Reduction for Constrained Bayesian Optimisation

Implementation of the method described in [Autoencoder-enhanced joint dimensionality reduction for constrained Bayesian optimisation](https://iopscience.iop.org/article/10.1088/2632-2153/ae0efe) (DOI: 10.1088/2632-2153/ae0efe)

## Quick summary
- Implements AeroBO and utilities to run experiments.
- Experiments and plotting assume PyTorch / GPyTorch / BoTorch ecosystem.

## Requirements
Recommended Python environment (tested on macOS / Linux):
- Python 3.8+
- PyTorch
- gpytorch
- botorch
- numpy
- scipy
- matplotlib
- pandas
- joblib
- scikit-learn

## Project layout (important files / folders)
- `main_aerobo.py` — entry points for running the aerobo algorithm.
- `aerobo_v1.py` — AeroBO algorithm implementation.
- `models_aerobo.py`, `gps.py` — model / GP helpers.
- `sampling.py` — sampling utilities.

## How to run experiment
```bash
python main_aerobo.py    # run AeroBO experiment
```
