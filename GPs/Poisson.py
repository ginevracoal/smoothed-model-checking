import os
import sys
import torch
import gpytorch
import numpy as np
import pandas as pd
import seaborn as sns
import pickle5 as pickle
import matplotlib.pyplot as plt
import torch.utils.data as data_utils

from variational_GP import GPmodel, train_GP 
from binomial_likelihood import BinomialLikelihood


def build_dataframe(data):
    params = torch.tensor(data['params'], dtype=torch.float32)
    labels = torch.tensor(data['labels'], dtype=torch.int)

    n_params = params.shape[1]
    n_trials = labels.shape[1]

    success_counts = [len(row[row==1.]) for row in labels]
    success_counts = torch.tensor(success_counts, dtype=torch.float32)

    print("\nparams shape =", params.shape)
    print("labels shape =", labels.shape)
    print("n. trials =", n_trials)
    # print("\nParams True label counts:\n", success_counts)

    return params, success_counts, n_params, n_trials

def satisfaction_function(lam):
    return torch.exp(torch.tensor(-lam))*(1+lam+(lam**2)/2+(lam**3)/6)

for train_filename in [
    "Poisson_DS_46samples_1obs_Lambda", 
    "Poisson_DS_46samples_5obs_Lambda",
    "Poisson_DS_46samples_10obs_Lambda",
    ]:

    print(f"\n=== Training {train_filename} ===")

    with open(f"../Data/Poisson/{train_filename}.pickle", 'rb') as handle:
        data = pickle.load(handle)
    x_train, y_train, n_params, n_trials = build_dataframe(data)

    n_trials_train = n_trials

    model = GPmodel(inducing_points=x_train)
    likelihood = BinomialLikelihood()

    model = train_GP(model=model, likelihood=likelihood, x_train=x_train, y_train=y_train, num_epochs=100)
    os.makedirs(os.path.dirname('models/'), exist_ok=True)
    torch.save(model.state_dict(), f'models/gp_state_{train_filename}.pth')

    print(f"\n=== Validation ===")

    state_dict = torch.load(f'models/gp_state_{train_filename}.pth')
    model.load_state_dict(state_dict)

    model.eval()    
    likelihood.eval()

    with torch.no_grad():

        x_test = []
        for col_idx in range(n_params):
            x_test.append(torch.linspace(0.1, 5, 100))
        x_test = torch.stack(x_test, dim=1)

        observed_pred = likelihood(model(x_test), n_trials=n_trials) 
        pred_samples = observed_pred.sample(sample_shape=torch.Size((1000,)))

        print("\npred_samples.shape =", pred_samples.shape, "= (n. binomial samples, n. test params)")
        pred_labels = pred_samples.mean(dim=0)
        pred_variance = pred_samples.var(dim=0)

    path='plots/Poisson/'
    os.makedirs(os.path.dirname(path), exist_ok=True)
    
    fig, ax = plt.subplots(1, n_params, figsize=(6*n_params, 5))

    for col_idx in range(n_params):
        single_param_x_train = x_train[:,col_idx]
        single_param_x_test = x_test[:,col_idx]

        axis = ax if n_params==1 else ax[col_idx]
        sns.scatterplot(x=single_param_x_train, y=y_train/n_trials_train, ax=axis,  
            label='training points')
        sns.lineplot(x=single_param_x_test, y=satisfaction_function(single_param_x_test), ax=axis, 
            label='true satisfaction')
        sns.lineplot(x=single_param_x_test, y=pred_labels/n_trials, ax=axis, label='pred satisfaction')
        axis.fill_between(single_param_x_test.numpy(), 
            (pred_labels-pred_variance)/n_trials, (pred_labels+pred_variance)/n_trials, alpha=0.5)
    
    fig.savefig(path+f"lineplot_{train_filename}.png")
    plt.close()


