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
from gpytorch.functions import log_normal_cdf
from variational_GP import GPmodel, train_GP 
from data_utils import build_bernoulli_dataframe, build_binomial_dataframe


parser = argparse.ArgumentParser()
parser.add_argument("--variational_distribution", default='cholesky', type=str, help="Variational distribution")
parser.add_argument("--variational_strategy", default='default', type=str, help="Variational strategy")
parser.add_argument("--train", default=True, type=eval, help="If True train the model else load it")
parser.add_argument("--n_epochs", default=1000, type=int, help="Number of training iterations")
parser.add_argument("--lr", default=0.01, type=float, help="Learning rate")
parser.add_argument("--n_test_points", default=100, type=int, help="Number of test params")
parser.add_argument("--n_posterior_samples", default=500, type=int, help="Number of samples from posterior distribution")
args = parser.parse_args()


def satisfaction_function(lam):
    lam = lam.clone().detach()
    return torch.exp(-lam)*(1+lam+(lam**2)/2+(lam**3)/6)


for train_filename in [
    "Poisson_DS_46samples_1obs_lambda", 
    "Poisson_DS_46samples_5obs_lambda",
    "Poisson_DS_46samples_10obs_lambda",
    ]:

    print(f"\n=== Training {train_filename} ===")

    out_filename = f"bernoulli_{train_filename}_epochs={args.n_epochs}_lr={args.lr}"

    with open(f"../Data/WorkingDatasets/Poisson/{train_filename}.pickle", 'rb') as handle:
        data = pickle.load(handle)
    x_train, y_train, n_params = build_bernoulli_dataframe(data)

    x_train_bin, y_train_bin, _, n_trials_train = build_binomial_dataframe(data)

    inducing_points = torch.tensor(data['params'], dtype=torch.float32)

    model = GPmodel(inducing_points=inducing_points, variational_distribution=args.variational_distribution,
        variational_strategy=args.variational_strategy)
    likelihood = gpytorch.likelihoods.BernoulliLikelihood()

    if args.train:

        model = train_GP(model=model, likelihood=likelihood, x_train=x_train, y_train=y_train, n_epochs=args.n_epochs, 
            lr=args.lr)
        os.makedirs(os.path.dirname("models/"), exist_ok=True)
        torch.save(model.state_dict(), "models/gp_state_"+out_filename+".pth")

    print(f"\n=== Validation ===")

    model.eval()    
    likelihood.eval()

    state_dict = torch.load("models/gp_state_"+out_filename+".pth")
    model.load_state_dict(state_dict)

    with torch.no_grad():

        x_test = []
        for col_idx in range(n_params):
            x_test.append(torch.linspace(0.1, 5, 100))
        x_test = torch.stack(x_test, dim=1)

        posterior_binomial = model(x_test)
        post_samples = posterior_binomial.sample(sample_shape=torch.Size((args.n_posterior_samples,)))
        pred_samples = [[torch.exp(log_normal_cdf(post_samples[j, i])) for i in range(args.n_test_points)] for j in range(args.n_posterior_samples)]
        pred_samples = torch.tensor(pred_samples)
        z = 1.96
        mu = torch.mean(pred_samples, dim=0)
        sigma = torch.std(pred_samples, dim=0)

    path='plots/Poisson/'
    os.makedirs(os.path.dirname(path), exist_ok=True)

    fig, ax = plt.subplots(1, n_params, figsize=(6*n_params, 5))

    for col_idx in range(n_params):
        single_param_x_train = x_train_bin[:,col_idx]
        single_param_x_test = x_test[:,col_idx]

        axis = ax if n_params==1 else ax[col_idx]

        sns.lineplot(x=single_param_x_test.numpy(), y=mu, ax=axis, label='posterior')
        axis.fill_between(single_param_x_test.numpy(), mu-z*sigma, mu+z*sigma, alpha=0.5)

        sns.lineplot(x=single_param_x_test, y=satisfaction_function(single_param_x_test), ax=axis, 
            label='true satisfaction')
        sns.scatterplot(x=single_param_x_train, y=y_train_bin.flatten()/n_trials_train, ax=axis, 
            label='training points', marker='.', color='black')
        

    fig.savefig(path+f"bernoulli_"+out_filename+".png")
    plt.close()


