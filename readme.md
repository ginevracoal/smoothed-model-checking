
## Project structure

- `Datasets/` contains the code to generate training and validation sets
    - `Models/` contains the specs of the CRN model and the config setting for the experiments
    - `Data_Generation/` generates dataset of pairs `(pameter, labels)` by generating CRN trajectories with respective STL boolean label. `labels` is a vector of length M of 0s and 1s, where M is the number of samples per parameter value.
    - `Data_Validation/` labels CRN trajectories wrt a STL requirement
    - `Data/WorkingDatasets` contains datasets and visualization plots
- `BNNs/` implements the Bayesian Neural Network model
- `GPs/` implements the Gaussian Process model

Datasets are loaded here: https://mega.nz/folder/pbBFnYSJ#XXwzMHfJaV4G9xKx3NMIwg

## Setup

Python version 3.7.6

Install virtual environment:
```
python -m venv venv
pip install -r requirements.txt
```

## Experiments

Activate the environment
```
source venv/bin/activate
```

In `Datasets/Data_Generation/run_data_generation.py` set `model_name` and `latin_flag` (to use a latin hypercube sampling strategy) and the run: `python run_data_generation.py`

Train models
```
python BNNs/train_bnn.py
python GPs/train_gp.py
```