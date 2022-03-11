import os
import sys
import torch
import gpytorch
import argparse
import numpy as np
import pandas as pd
import seaborn as sns
from math import sqrt
import pickle5 as pickle
import matplotlib.pyplot as plt

from variational_GP import GPmodel, train_GP 
from data_utils import build_bernoulli_dataframe, build_binomial_dataframe


parser = argparse.ArgumentParser()
parser.add_argument("--train", default=True, type=eval, help="If True train the model else load it")
parser.add_argument("--n_epochs", default=1000, type=int, help="Number of training iterations")
parser.add_argument("--lr", default=0.01, type=int, help="Learning rate")
parser.add_argument("--n_test_points", default=100, type=int, help="Number of test params")
args = parser.parse_args()


for train_filename, val_filename in [
    ["SIR_DS_200samples_10obs_Beta", "SIR_DS_20samples_5000obs_Beta"],
    ["SIR_DS_200samples_10obs_Gamma", "SIR_DS_20samples_5000obs_Gamma"],
    ["SIR_DS_256samples_5000obs_BetaGamma", "SIR_DS_256samples_10obs_BetaGamma"]
    ]:

    print(f"\n=== Training {train_filename} ===")

    out_filename = f"{train_filename}_epochs={args.n_epochs}_lr={args.lr}"

    with open(f"../Data/SIR/{train_filename}.pickle", 'rb') as handle:
        data = pickle.load(handle)
    x_train, y_train, n_params = build_bernoulli_dataframe(data)

    inducing_points = torch.tensor(data['params'], dtype=torch.float32)

    model = GPmodel(inducing_points=inducing_points)
    likelihood = gpytorch.likelihoods.BernoulliLikelihood()

    if args.train:

        model = train_GP(model=model, likelihood=likelihood, x_train=x_train, y_train=y_train, n_epochs=args.n_epochs, 
            lr=args.lr)
        os.makedirs(os.path.dirname("models/"), exist_ok=True)
        torch.save(model.state_dict(), "models/gp_state_"+out_filename+".pth")

    print(f"\n=== Validation {val_filename} ===")

    model.eval()    
    likelihood.eval()

    with open(f"../Data/SIR/{val_filename}.pickle", 'rb') as handle:
        data = pickle.load(handle)
    x_val, y_val, n_params, n_trials = build_binomial_dataframe(data)

    model = GPmodel(inducing_points=inducing_points)
    state_dict = torch.load("models/gp_state_"+out_filename+".pth")
    model.load_state_dict(state_dict)

    likelihood = gpytorch.likelihoods.BernoulliLikelihood()

    with torch.no_grad():

        x_test = []
        for col_idx in range(n_params):
            single_param_values = x_val[:,col_idx]
            x_test.append(torch.linspace(single_param_values.min(), single_param_values.max(), args.n_test_points))
        x_test = torch.stack(x_test, dim=1)

        n_bernoulli_samples = 1000
        posterior_bernoulli = likelihood(model(x_test)) 
        pred_samples = posterior_bernoulli.sample(sample_shape=torch.Size((n_bernoulli_samples,)))

        # print("\npred_samples.shape =", pred_samples.shape, "= (n. bernoulli samples, n. test params)")

    # z = 1.96
    # pred_mean = pred_samples.mean(0)
    # pred_std = pred_samples.std(0)
    # lower_ci = pred_mean-z*pred_std/sqrt(n_bernoulli_samples)
    # upper_ci = pred_mean+z*pred_std/sqrt(n_bernoulli_samples)
 
    path='plots/SIR/'
    os.makedirs(os.path.dirname(path), exist_ok=True)

    fig, ax = plt.subplots(1, n_params, figsize=(6*n_params, 5))

    for col_idx in range(n_params):
        single_param_x_val = x_val[:,col_idx]
        single_param_x_test = x_test[:,col_idx]

        axis = ax if n_params==1 else ax[col_idx]
        sns.scatterplot(x=single_param_x_val, y=y_val/n_trials, ax=axis, label='validation pts')

        sns.lineplot(x=single_param_x_test, y=posterior_bernoulli.mean, ax=axis, label='posterior')
        axis.fill_between(single_param_x_test.numpy(), posterior_bernoulli.mean-posterior_bernoulli.variance, 
            posterior_bernoulli.mean+posterior_bernoulli.variance, alpha=0.5)

    fig.savefig(path+f"bernoulli_"+out_filename+".png")
    plt.close()


