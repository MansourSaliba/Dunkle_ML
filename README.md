# Dunkle_ML

This repository contains the scripts and data artifacts used to reproduce Phase I and Phase II modeling results for the Dunkle solar still system. The layout is scripts-first and organized by phase, with synthetic data included and experimental data documented as restricted.

## Repository Layout

- [data/](data/) raw/interim/processed data folders with details in [data/raw/README.md](data/raw/README.md), [data/interim/README.md](data/interim/README.md), and [data/processed/README.md](data/processed/README.md)
- [docs/](docs/) project report and presentation PDFs
- [results/](results/) exported figures used in the writeup
- [scripts/](scripts/) all modeling and visualization scripts
- [scripts/Synthetic%20data/dunkle_forward_model.py](scripts/Synthetic%20data/dunkle_forward_model.py) synthetic data generation
- [scripts/Phase%20I%20-%20hc%20and%20he%20discovery/](scripts/Phase%20I%20-%20hc%20and%20he%20discovery/) Phase I discovery methods
- [scripts/Phase%20II%20-%20ODE%20discovery/](scripts/Phase%20II%20-%20ODE%20discovery/) Phase II ODE discovery methods
- [scripts/Visualization/plots_merged.m](scripts/Visualization/plots_merged.m) MATLAB figure regeneration
- [DATA.md](DATA.md) data access policy and restrictions
- [requirements.txt](requirements.txt) Python dependencies

## Data Policy

Synthetic data is provided under [data/processed/](data/processed). Experimental data is restricted and documented in [DATA.md](DATA.md). The scripts that require experimental data will not run without the private workbook described there.

## How To Reproduce The Work

1. Clone the repository and move into the repo root.
2. Create and activate a Python environment, then install dependencies listed in [requirements.txt](requirements.txt).
3. (Optional) Regenerate synthetic data by running [scripts/Synthetic%20data/dunkle_forward_model.py](scripts/Synthetic%20data/dunkle_forward_model.py).
4. Run Phase I discovery scripts (hc and he).
	- Baseline NN: [scripts/Phase%20I%20-%20hc%20and%20he%20discovery/Baseline%20NN/Phase_I_NN.py](scripts/Phase%20I%20-%20hc%20and%20he%20discovery/Baseline%20NN/Phase_I_NN.py)
	- SINDy Trials: [Trial 1](scripts/Phase%20I%20-%20hc%20and%20he%20discovery/SINDy%20-%20Trial%201/Phase_I_SINDy.py), [Trial 2](scripts/Phase%20I%20-%20hc%20and%20he%20discovery/SINDy%20-%20Trial%202/Phase_I_SINDy_Trial_2.py), [Trial 3](scripts/Phase%20I%20-%20hc%20and%20he%20discovery/SINDy%20-%20Trial%203/Phase_I_SINDy_Trial_3.py)
	- PySR: [scripts/Phase%20I%20-%20hc%20and%20he%20discovery/pySR/Phase_I_SR.py](scripts/Phase%20I%20-%20hc%20and%20he%20discovery/pySR/Phase_I_SR.py)
5. Run Phase II ODE discovery scripts (synthetic and experimental).
	- SINDy synthetic: [scripts/Phase%20II%20-%20ODE%20discovery/SINDy%20-%20Synthetic/Phase_II_SINDy.py](scripts/Phase%20II%20-%20ODE%20discovery/SINDy%20-%20Synthetic/Phase_II_SINDy.py)
	- SINDy experimental: [scripts/Phase%20II%20-%20ODE%20discovery/SINDy%20-%20Experimental/Phase_II_SINDy_exp.py](scripts/Phase%20II%20-%20ODE%20discovery/SINDy%20-%20Experimental/Phase_II_SINDy_exp.py)
	- PySR: [scripts/Phase%20II%20-%20ODE%20discovery/pySR/Phase_II_pySR.py](scripts/Phase%20II%20-%20ODE%20discovery/pySR/Phase_II_pySR.py)
	- PINN: [scripts/Phase%20II%20-%20ODE%20discovery/PINN/Phase_II_PINN.py](scripts/Phase%20II%20-%20ODE%20discovery/PINN/Phase_II_PINN.py)
6. (Optional) Rebuild figures in MATLAB using [scripts/Visualization/plots_merged.m](scripts/Visualization/plots_merged.m).

Each script writes its outputs (CSV, Excel, and PNG) into the same folder as the script unless stated otherwise inside the script.