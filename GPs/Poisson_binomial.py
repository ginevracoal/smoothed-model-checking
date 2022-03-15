import os
import sys
import torch
import gpytorch
import argparse
import numpy as np
import pandas as pd
import seaborn as sns
import pickle5 as pickle
import matplotlib.pyplot as plt
import torch.utils.data as data_utils

from variational_GP import GPmodel, train_GP 
from data_utils import build_binomial_dataframe
from binomial_likelihood import BinomialLikelihood

# from Poisson_bernoulli import satisfaction_function


parser = argparse.ArgumentParser()
parser.add_argument("--variational_distribution", default='cholesky', type=str, help="Variational distribution")
parser.add_argument("--variational_strategy", default='default', type=str, help="Variational strategy")
parser.add_argument("--train", default=True, type=eval, help="If True train the model else load it")
parser.add_argument("--n_epochs", default=1000, type=int, help="Number of training iterations")
parser.add_argument("--lr", default=0.01, type=float, help="Learning rate")
parser.add_argument("--n_test_points", default=100, type=int, help="Number of test params")
parser.add_argument("--n_posterior_samples", default=1000, type=int, help="Number of samples from posterior distribution")
args = parser.parse_args()

def satisfaction_function(lam):
    return torch.exp(-lam)*(1+lam+(lam**2)/2+(lam**3)/6)


for train_filename in [
    "Poisson_DS_46samples_1obs_Lambda", 
    "Poisson_DS_46samples_5obs_Lambda",
    "Poisson_DS_46samples_10obs_Lambda",
    ]:

    print(f"\n=== Training {train_filename} ===")

    out_filename = f"binomial_{train_filename}_epochs={args.n_epochs}_lr={args.lr}"

    with open(f"../Data/Poisson/{train_filename}.pickle", 'rb') as handle:
        data = pickle.load(handle)
    x_train, y_train, n_params, n_trials_train = build_binomial_dataframe(data)

    model = GPmodel(inducing_points=x_train, variational_distribution=args.variational_distribution,
        variational_strategy=args.variational_strategy)
    likelihood = BinomialLikelihood()
    likelihood.n_trials = n_trials_train

    if args.train:

        model = train_GP(model=model, likelihood=likelihood, x_train=x_train, y_train=y_train, n_epochs=args.n_epochs, 
            lr=args.lr)
        os.makedirs(os.path.dirname('models/'), exist_ok=True)
        torch.save(model.state_dict(), f"models/gp_state_{out_filename}.pth")

    print(f"\n=== Validation ===")

    model.eval()    
    likelihood.eval()
    likelihood.n_trials = 1

    state_dict = torch.load(f'models/gp_state_{out_filename}.pth')
    model.load_state_dict(state_dict)

    with torch.no_grad():

        x_test = []
        for col_idx in range(n_params):
            x_test.append(torch.linspace(0.1, 5, args.n_test_points))
        x_test = torch.stack(x_test, dim=1)

        posterior_binomial = likelihood(model(x_test))

        # pred_samples = posterior_binomial.sample(sample_shape=torch.Size((args.n_posterior_samples,)))
        # print("\npred_samples.shape =", pred_samples.shape, "= (n. binomial samples, n. test params)")

    path='plots/Poisson/'
    os.makedirs(os.path.dirname(path), exist_ok=True)
    
    fig, ax = plt.subplots(1, n_params, figsize=(6*n_params, 5))

    mu = posterior_binomial.mean
    sigma = posterior_binomial.variance

    for col_idx in range(n_params):
        single_param_x_train = x_train[:,col_idx]
        single_param_x_test = x_test[:,col_idx]

        axis = ax if n_params==1 else ax[col_idx]

        sns.lineplot(x=single_param_x_test.numpy(), y=mu, ax=axis, label='posterior')
        axis.fill_between(single_param_x_test.numpy(), mu-sigma, mu+sigma, alpha=0.5)

        sns.lineplot(x=single_param_x_test, y=satisfaction_function(single_param_x_test), ax=axis, 
            label='true satisfaction')
        sns.scatterplot(x=single_param_x_train, y=y_train/n_trials_train, ax=axis, 
            label='training points', marker='.', color='black')
    
    fig.savefig(path+f"binomial_{out_filename}.png")
    plt.close()


