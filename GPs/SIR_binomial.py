import os
import sys
import torch
import gpytorch
import argparse
import numpy as np
import pandas as pd
import seaborn as sns
from math import sqrt
import pickle
import matplotlib.pyplot as plt
import torch.utils.data as data_utils
from gpytorch.functions import log_normal_cdf
from variational_GP import GPmodel, train_GP 
from data_utils import build_binomial_dataframe
from binomial_likelihood import BinomialLikelihood


parser = argparse.ArgumentParser()
parser.add_argument("--variational_distribution", default='cholesky', type=str, help="Variational distribution")
parser.add_argument("--variational_strategy", default='default', type=str, help="Variational strategy")
parser.add_argument("--train", default=True, type=eval, help="If True train the model else load it")
parser.add_argument("--n_epochs", default=1000, type=int, help="Number of training iterations")
parser.add_argument("--lr", default=0.01, type=float, help="Learning rate")
parser.add_argument("--n_test_points", default=100, type=int, help="Number of test params")
parser.add_argument("--n_posterior_samples", default=500, type=int, help="Number of samples from posterior distribution")
args = parser.parse_args()


for train_filename, val_filename in [
    ["SIR_DS_200samples_10obs_beta", "SIR_DS_20samples_5000obs_beta"],
     ["SIR_DS_200samples_10obs_gamma", "SIR_DS_20samples_5000obs_gamma"],
    # ["SIR_DS_256samples_5000obs_betagamma", "SIR_DS_256samples_10obs_betagamma"]
    ]:

    print(f"\n=== Training {train_filename} ===")

    out_filename = f"binomial_{train_filename}_epochs={args.n_epochs}_lr={args.lr}"

    with open(f"../Data/WorkingDatasets/SIR/{train_filename}.pickle", 'rb') as handle:
        data = pickle.load(handle)
    x_train, y_train, n_params, n_trials_train = build_binomial_dataframe(data)
    inducing_points = torch.tensor(data['params'], dtype=torch.float32)

    model = GPmodel(inducing_points=inducing_points, variational_distribution=args.variational_distribution,
        variational_strategy=args.variational_strategy)
    likelihood = BinomialLikelihood()
    likelihood.n_trials = n_trials_train

    if args.train:

        model = train_GP(model=model, likelihood=likelihood, x_train=x_train, y_train=y_train, n_epochs=args.n_epochs, 
            lr=args.lr)
        os.makedirs(os.path.dirname('models/'), exist_ok=True)
        torch.save(model.state_dict(), f"models/gp_state_{out_filename}.pth")

    print(f"\n=== Validation {val_filename} ===")

    model.eval()    
    likelihood.eval()

    with open(f"../Data/WorkingDatasets/SIR/{val_filename}.pickle", 'rb') as handle:
        data = pickle.load(handle)
    x_val, y_val, n_params, n_trials_val = build_binomial_dataframe(data)

    state_dict = torch.load(f'models/gp_state_{out_filename}.pth')
    model.load_state_dict(state_dict)

    with torch.no_grad():

        x_test = []
        for col_idx in range(n_params):
            single_param_values = x_val[:,col_idx]
            x_test.append(torch.linspace(single_param_values.min(), single_param_values.max(), args.n_test_points))
        x_test = torch.stack(x_test, dim=1)

        if n_params==2:
            x_test = torch.tensor(list(product(x_test[:,0], x_test[:,1])))

        #posterior_binomial = likelihood(model(x_test))
        posterior_binomial = model(x_test)
        post_samples = posterior_binomial.sample(sample_shape=torch.Size((args.n_posterior_samples,)))
        pred_samples = [[torch.exp(log_normal_cdf(post_samples[j, i])) for i in range(args.n_test_points)] for j in range(args.n_posterior_samples)]
        pred_samples = torch.tensor(pred_samples)
        z = 1.96
        mu = torch.mean(pred_samples, dim=0)
        sigma = torch.std(pred_samples, dim=0)
        
    path='plots/SIR/'
    os.makedirs(os.path.dirname(path), exist_ok=True)

    if n_params==1:

        fig, ax = plt.subplots(1, 1, figsize=(6*n_params, 5))

        sns.scatterplot(x=x_val.flatten(), y=y_val.flatten()/n_trials_val, ax=ax, label='validation pts')
        sns.scatterplot(x=x_train.flatten(), y=y_train.flatten()/n_trials_train, ax=ax, 
            label='training points', marker='.', color='black')

        sns.lineplot(x=x_test.flatten(), y=mu, ax=ax, label='posterior')
        ax.fill_between(x_test.flatten(), mu-z*sigma, mu+z*sigma, alpha=0.5)

    else:
        raise NotImplementedError

        # fig, ax = plt.subplots(1, n_params, figsize=(6*n_params, 5))

    fig.savefig(path+f"{out_filename}.png")
    plt.close()


