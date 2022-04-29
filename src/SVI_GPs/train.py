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
from posterior_plot_utils import plot_posterior
from SVI_GPs.variational_GP import GPmodel
from data_utils import normalize_columns, get_tensor_data


random.seed(0)
np.random.seed(0)
torch.manual_seed(0)

parser = argparse.ArgumentParser()
parser.add_argument("--likelihood", default='binomial', type=str, help="Choose 'bernoulli' or 'binomial'")
parser.add_argument("--variational_distribution", default='cholesky', type=str, help="Variational distribution: cholesky, meanfield")
parser.add_argument("--variational_strategy", default='default', type=str, help="Variational strategy: default, unwhitened, batchdecoupled")
parser.add_argument("--load", default=False, type=eval, help="If True load the model else train it")
parser.add_argument("--batch_size", default=100, type=int, help="Batch size")
parser.add_argument("--n_epochs", default=1000, type=int, help="Max number of training iterations")
parser.add_argument("--lr", default=0.01, type=float, help="Learning rate")
parser.add_argument("--n_posterior_samples", default=100, type=int, help="Number of samples from posterior distribution")
args = parser.parse_args()
print(args)

plots_path = os.path.join(plots_path, "SVI_GPs/")
models_path = os.path.join(models_path, "SVI_GPs/")

for filepath, train_filename, val_filename, params_list, math_params_list in case_studies:

    if len(params_list)==6:
        args.epochs = 100
        args.batch_size = 5000

    print(f"\n=== SVI GP Training {train_filename} ===")

    out_filename = f"svi_gp_{train_filename}_epochs={args.n_epochs}_lr={args.lr}_batch={args.batch_size}_{args.variational_distribution}_{args.variational_strategy}"

    with open(os.path.join(data_path, filepath, train_filename+".pickle"), 'rb') as handle:
        train_data = pickle.load(handle)

    inducing_points = normalize_columns(get_tensor_data(train_data)[0])

    model = GPmodel(inducing_points=inducing_points, variational_distribution=args.variational_distribution,
        variational_strategy=args.variational_strategy, likelihood=args.likelihood)

    if args.load:
        model.load(filepath=models_path, filename=out_filename)

    else:
        model.train_gp(train_data=train_data, n_epochs=args.n_epochs, lr=args.lr, batch_size=args.batch_size)
        model.save(filepath=models_path, filename=out_filename)

    print(f"\n=== SVI GP Validation {val_filename} ===")

    with open(os.path.join(data_path, filepath, val_filename+".pickle"), 'rb') as handle:
        val_data = pickle.load(handle)
    
    post_mean, q1, q2, evaluation_dict = model.evaluate(train_data=train_data, val_data=val_data, 
        n_posterior_samples=args.n_posterior_samples)

    if len(params_list)<=2:

        fig = plot_posterior(params_list=params_list, math_params_list=math_params_list, train_data=train_data,
            test_data=val_data, val_data=val_data, post_mean=post_mean, q1=q1, q2=q2)

        os.makedirs(os.path.dirname(plots_path), exist_ok=True)
        fig.savefig(plots_path+f"{out_filename}.png")

