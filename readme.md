
## Project structure

- `Datasets/` contains the code to generate training and validation sets
    - `Models/` contains the specs of the CRN model and the config setting for the experiments
    - `Data_Generation/` generates dataset of pairs `(pameter, labels)` by generating CRN trajectories with respective STL boolean label. `labels` is a vector of length M of 0s and 1s, where M is the number of samples per parameter value.
    - `Data_Validation/` labels CRN trajectories wrt a STL requirement
    - `Data/WorkingDatasets` contains datasets and visualization plots
- `BNNs/` implements the Bayesian Neural Network model
- `GPs/` implements the Gaussian Process model

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

### Download training and validation datasets

Download the available datasets in a folder `src/data/` from: https://mega.nz/folder/pbBFnYSJ#XXwzMHfJaV4G9xKx3NMIwg

Time needed to generate the data: https://www.dropbox.com/scl/fi/h3twljfitial1galoui9n/Datasets.paper?dl=0&rlkey=625cm5u25h1d4qqnwn1jbuw4g

To generate new datasets update the configuration files `src/Datasets/Data_Generation/*_config.py` and run `python efficient_data_gen.py` with the desired settings. 

### Case studies

`src/paths.py` contains the informations needed to perform training and evaluation on the several case studies.
Comment out the unwanted lines to exclude them from computations.

### Train and evaluate

Train and evaluate EP GP, SVI GP and SVI BNN models:
```
python EP_GPs/train_bnn.py
python SVI_GPs/train_gp.py
python SVI_BNNs/train_bnn.py
```

Plot final comparison between the trained models and get summary statistics:
```
python plot_comparison.py
```

Trained models are saved in `src/out/models/`, executions logs are saved in `src/out/logs/`, plots are saved in `src/out/plots/`, summary statistic are reported in `src/out/plots/comparison_table.txt`.

To reproduce plots from the paper simply run `./exec.sh`.
