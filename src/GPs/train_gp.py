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

sys.path.append(".")
from paths import *
from plot_utils import plot_posterior
from GPs.variational_GP import GPmodel
from data_utils import normalize_columns, get_tensor_data


random.seed(0)
np.random.seed(0)
torch.manual_seed(0)

parser = argparse.ArgumentParser()
parser.add_argument("--likelihood", default='binomial', type=str, help='Choose bernoulli or binomial')
parser.add_argument("--variational_distribution", default='cholesky', type=str, help="Variational distribution")
parser.add_argument("--variational_strategy", default='unwhitened', type=str, help="Variational strategy")
parser.add_argument("--load", default=False, type=eval, help="If True load the model else train it")
parser.add_argument("--n_epochs", default=1000, type=int, help="Max number of training iterations")
parser.add_argument("--lr", default=0.01, type=float, help="Learning rate")
parser.add_argument("--n_posterior_samples", default=10, type=int, help="Number of samples from posterior distribution")
args = parser.parse_args()

models_path = os.path.join("GPs", models_path)
plots_path = os.path.join("GPs", plots_path)
os.makedirs(os.path.dirname(models_path), exist_ok=True)
os.makedirs(os.path.dirname(plots_path), exist_ok=True)


for filepath, train_filename, val_filename, params_list, math_params_list in data_paths:

    print(f"\n=== Training {train_filename} ===")

    with open(os.path.join(data_path, filepath, train_filename+".pickle"), 'rb') as handle:
        train_data = pickle.load(handle)

    n_epochs = args.n_epochs
    out_filename = f"{args.likelihood}_{train_filename}_epochs={n_epochs}_lr={args.lr}"
    inducing_points = normalize_columns(get_tensor_data(train_data)[0])

    model = GPmodel(inducing_points=inducing_points, variational_distribution=args.variational_distribution,
        variational_strategy=args.variational_strategy, likelihood=args.likelihood)

    if args.load:
        model.load(filepath=models_path, filename=out_filename)

    else:
        model.train_GP(train_data=train_data, n_epochs=n_epochs, lr=args.lr)
        model.save(filepath=models_path, filename=out_filename)

    print(f"\n=== Validation {val_filename} ===")

    if filepath=='Poisson':

        raise NotImplementedError

        # x_test, post_samples, post_mean, post_std, evaluation_dict = evaluate_GP(model=model, likelihood=likelihood,
        #     n_posterior_samples=args.n_posterior_samples, n_params=n_params)

        # fig = plot_GP_posterior(case_study=filepath, x_train_binomial=x_train_binomial, y_train_binomial=y_train_binomial, 
        #     n_trials_train=n_trials_train, x_test=x_test, post_mean=post_mean, post_std=post_std, params_list=params_list)
        # os.makedirs(os.path.dirname(plots_path), exist_ok=True)
        # fig.savefig(plots_path+f"{out_filename}.png")

    else: 

        with open(os.path.join(data_path, filepath, val_filename+".pickle"), 'rb') as handle:
            val_data = pickle.load(handle)
        
        post_mean, q1, q2, evaluation_dict = model.eval_GP(val_data=val_data, n_posterior_samples=args.n_posterior_samples)

        if len(params_list)<=2:

            fig = plot_posterior(params_list=params_list, math_params_list=math_params_list, train_data=train_data,
                test_data=val_data, val_data=val_data, post_mean=post_mean, q1=q1, q2=q2)

            os.makedirs(os.path.dirname(plots_path), exist_ok=True)
            fig.savefig(plots_path+f"{out_filename}.png")
