
## Project structure

- `Datasets/` contains the code to generate training and validation sets
    - `Models/` contains the specs of the CRN model and the config setting for the experiments
    - `Data_Generation/` generates CRN trajectories
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
