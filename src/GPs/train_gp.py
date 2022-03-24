import os
import sys
import torch
import random
import gpytorch
import argparse
import numpy as np
from math import sqrt
import pickle5 as pickle
import matplotlib.pyplot as plt

from variational_GP import GPmodel, train_GP, evaluate_GP, plot_GP_posterior
from bernoulli_likelihood import BernoulliLikelihood
from binomial_likelihood import BinomialLikelihood
from utils import build_bernoulli_dataframe, build_binomial_dataframe, normalize_columns

sys.path.append('../')
from paths import *

random.seed(0)
np.random.seed(0)
torch.manual_seed(0)

parser = argparse.ArgumentParser()
parser.add_argument("--likelihood", default='binomial', type=str, help='Choose bernoulli or binomial')
parser.add_argument("--variational_distribution", default='cholesky', type=str, help="Variational distribution")
parser.add_argument("--variational_strategy", default='unwhitened', type=str, help="Variational strategy")
parser.add_argument("--load", default=False, type=eval, help="If True load the model else train it")
parser.add_argument("--n_epochs", default=1000, type=int, help="Number of training iterations")
parser.add_argument("--lr", default=0.01, type=float, help="Learning rate")
parser.add_argument("--n_test_points", default=30, type=int, help="Number of test params")
parser.add_argument("--n_posterior_samples", default=10, type=int, help="Number of samples from posterior distribution")
args = parser.parse_args()


for filepath, train_filename, val_filename, params_list in data_paths:

    print(f"\n=== Training {train_filename} ===")

    out_filename = f"{args.likelihood}_{train_filename}_epochs={args.n_epochs}_lr={args.lr}"

    with open(os.path.join(data_path, filepath, train_filename+".pickle"), 'rb') as handle:
        data = pickle.load(handle)

    if args.likelihood=='bernoulli':
        x_train, y_train, n_params = build_bernoulli_dataframe(data)
        x_train_binomial, y_train_binomial, _, n_trials_train = build_binomial_dataframe(data)
        likelihood = BernoulliLikelihood()

    elif args.likelihood=='binomial':
        x_train, y_train, n_params, n_trials_train = build_binomial_dataframe(data)
        x_train_binomial, y_train_binomial = x_train, y_train
        likelihood = BinomialLikelihood()

    else:
        raise NotImplementedError

    normalized_x_train = normalize_columns(x_train) 
    inducing_points = normalize_columns(x_train_binomial)

    model = GPmodel(inducing_points=inducing_points, variational_distribution=args.variational_distribution,
        variational_strategy=args.variational_strategy)

    if args.load:

        state_dict = torch.load(os.path.join(models_path, "gp_state_"+out_filename+".pth"))
        model.load_state_dict(state_dict)

    else:

        model = train_GP(model=model, likelihood=likelihood, x_train=normalized_x_train, 
            y_train=y_train, n_trials_train=n_trials_train, n_epochs=args.n_epochs, lr=args.lr)
        os.makedirs(os.path.dirname(models_path), exist_ok=True)
        torch.save(model.state_dict(), os.path.join(models_path, "gp_state_"+out_filename+".pth"))

    print(f"\n=== Validation {val_filename} ===")

    if filepath=='Poisson':

        evaluate_GP(model=model, likelihood=likelihood, n_test_points=args.n_test_points,
            n_posterior_samples=args.n_posterior_samples, n_params=n_params)  

        if n_params<=2:

            x_test, post_mean, post_std = evaluate_GP(model=model, likelihood=likelihood, 
                n_posterior_samples=args.n_posterior_samples, n_params=n_params, n_test_points=args.n_test_points)  

            fig = plot_GP_posterior(x_train_binomial=x_train_binomial, y_train_binomial=y_train_binomial, 
                n_trials_train=n_trials_train, x_test=x_test, post_mean=post_mean, post_std=post_std, 
                params_list=params_list)
            os.makedirs(os.path.dirname(plots_path), exist_ok=True)
            fig.savefig(plots_path+f"{out_filename}.png")

    else: 
        with open(os.path.join(data_path, filepath, val_filename+".pickle"), 'rb') as handle:
            val_data = pickle.load(handle)
        
        x_val, y_val, n_params, n_trials_val = build_binomial_dataframe(val_data)

        evaluate_GP(model=model, likelihood=likelihood, x_val=x_val, y_val=y_val, 
            n_trials_val=n_trials_val, n_posterior_samples=args.n_posterior_samples, n_params=n_params) 
        
        if n_params<=2:

            x_test, post_mean, post_std = evaluate_GP(model=model, likelihood=likelihood, x_val=x_val, y_val=y_val,
                n_trials_val=n_trials_val, n_posterior_samples=args.n_posterior_samples, n_params=n_params, 
                n_test_points=args.n_test_points)  

            fig = plot_GP_posterior(x_train_binomial=x_train_binomial, y_train_binomial=y_train_binomial, 
                n_trials_train=n_trials_train, x_test=x_test, post_mean=post_mean, post_std=post_std, 
                params_list=params_list, x_val=x_val, y_val=y_val, n_trials_val=n_trials_val)
            os.makedirs(os.path.dirname(plots_path), exist_ok=True)
            fig.savefig(plots_path+f"{out_filename}.png")
