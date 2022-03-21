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
from itertools import product
import matplotlib.pyplot as plt
from gpytorch.functions import log_normal_cdf

from variational_GP import GPmodel, train_GP 
from bernoulli_likelihood import BernoulliLikelihood
from binomial_likelihood import BinomialLikelihood
from utils import build_bernoulli_dataframe, build_binomial_dataframe, Poisson_satisfaction_function

data_path = '../../data/'
sys.path.append('../')
from paths import *

parser = argparse.ArgumentParser()
parser.add_argument("--likelihood", default='binomial', type=str, help='Choose bernoulli or binomial')
parser.add_argument("--variational_distribution", default='cholesky', type=str, help="Variational distribution")
parser.add_argument("--variational_strategy", default='unwhitened', type=str, help="Variational strategy")
parser.add_argument("--train", default=True, type=eval, help="If True train the model else load it")
parser.add_argument("--n_epochs", default=1000, type=int, help="Number of training iterations")
parser.add_argument("--lr", default=0.01, type=float, help="Learning rate")
parser.add_argument("--n_test_points", default=20, type=int, help="Number of test params")
parser.add_argument("--n_posterior_samples", default=500, type=int, help="Number of samples from posterior distribution")
args = parser.parse_args()


for filepath, train_filename, val_filename, params_list in data_paths:

    print(f"\n=== Training {train_filename} ===")

    out_filename = f"{args.likelihood}_{train_filename}_epochs={args.n_epochs}_lr={args.lr}"

    with open(os.path.join(data_path, filepath, train_filename+".pickle"), 'rb') as handle:
        data = pickle.load(handle)

    if args.likelihood=='bernoulli':
        x_train, y_train, n_params = build_bernoulli_dataframe(data)
        x_train_plot, y_train_plot, _, n_trials_train = build_binomial_dataframe(data)
        likelihood = BernoulliLikelihood()

    elif args.likelihood=='binomial':
        x_train, y_train, n_params, n_trials_train = build_binomial_dataframe(data)
        x_train_plot, y_train_plot = x_train, y_train
        likelihood = BinomialLikelihood()
        likelihood.n_trials = n_trials_train

    else:
        raise NotImplementedError

    inducing_points = torch.tensor(data['params'], dtype=torch.float32)

    model = GPmodel(inducing_points=inducing_points, variational_distribution=args.variational_distribution,
        variational_strategy=args.variational_strategy)

    if args.train:

        model = train_GP(model=model, likelihood=likelihood, x_train=x_train, y_train=y_train, 
            n_epochs=args.n_epochs, lr=args.lr)
        os.makedirs(os.path.dirname(models_path), exist_ok=True)
        torch.save(model.state_dict(), os.path.join(models_path, "gp_state_"+out_filename+".pth"))

    model.eval()    
    likelihood.eval()

    state_dict = torch.load(os.path.join(models_path, "gp_state_"+out_filename+".pth"))
    model.load_state_dict(state_dict)

    with torch.no_grad():

        if filepath=='Poisson':

            print(f"\n=== Validation ===")

            x_test = []
            for col_idx in range(n_params):
                x_test.append(torch.linspace(0.1, 5, args.n_test_points))
            x_test = torch.stack(x_test, dim=1)

        else:

            print(f"\n=== Validation {val_filename} ===")

            with open(os.path.join(data_path, filepath, val_filename+".pickle"), 'rb') as handle:
                data = pickle.load(handle)
            x_val, y_val, n_params, n_trials_val = build_binomial_dataframe(data)

            x_test = []
            for col_idx in range(n_params):
                single_param_values = x_val[:,col_idx]
                x_test.append(torch.linspace(single_param_values.min(), single_param_values.max(), args.n_test_points))
            x_test = torch.stack(x_test, dim=1)

            if n_params==2:
                x_test = torch.tensor(list(product(x_test[:,0], x_test[:,1])))

        posterior = model(x_test)
        post_samples = posterior.sample(sample_shape=torch.Size((args.n_posterior_samples,)))
        pred_samples = [[torch.exp(log_normal_cdf(post_samples[j, i])) for i in range(args.n_test_points)] for j in range(args.n_posterior_samples)]
        pred_samples = torch.tensor(pred_samples)
        
        z = 1.96
        mean = torch.mean(pred_samples, dim=0).flatten()
        std = torch.std(pred_samples, dim=0).flatten()
 
    os.makedirs(os.path.dirname(plots_path), exist_ok=True)

    if n_params==1:

        fig, ax = plt.subplots(1, 1, figsize=(8, 5))
        
        if filepath=="Poisson":

            sns.lineplot(x=x_test.flatten(), y=mean, ax=ax, label='posterior')
            ax.fill_between(x_test.flatten(), mean-z*std, mean+z*std, alpha=0.5)

            sns.lineplot(x=x_test.flatten(), y=Poisson_satisfaction_function(x_test).flatten(), ax=ax, 
                label='true satisfaction')
            sns.scatterplot(x=x_train_plot.flatten(), y=y_train_plot.flatten()/n_trials_train, ax=ax, 
                label='training points', marker='.', color='black')

        else:
            sns.scatterplot(x=x_val.flatten(), y=y_val/n_trials_val, ax=ax, label='validation pts')
            sns.scatterplot(x=x_train_plot.flatten(), y=y_train_plot/n_trials_train, ax=ax, 
                label='training points', marker='.', color='black')
            sns.lineplot(x=x_test.flatten(), y=mean, ax=ax, label='posterior')
            ax.fill_between(x_test.flatten(), mean-z*std, mean+z*std, alpha=0.5)

    elif n_params==2:

        p1, p2 = params_list

        fig, ax = plt.subplots(1, 2, figsize=(12, 5))

        data = pd.DataFrame({p1:x_val[:,0],p2:x_val[:,1],'val_counts':y_val.flatten()/n_trials_val})
        data[p1] = data[p1].apply(lambda x: format(float(x),".2f"))
        data[p2] = data[p2].apply(lambda x: format(float(x),".2f"))
        data = data.pivot(p1, p2, "val_counts")
        sns.heatmap(data, ax=ax[0], label='validation pts')

        data = pd.DataFrame({p1:x_test[:,0],p2:x_test[:,1],'posterior_mean':posterior.mean})
        data[p1] = data[p1].apply(lambda x: format(float(x),".2f"))
        data[p2] = data[p2].apply(lambda x: format(float(x),".2f"))
        data = data.pivot(p1, p2, "posterior_mean")
        sns.heatmap(data, ax=ax[1], label='posterior mean')

    plt.tight_layout()
    fig.savefig(plots_path+f"{out_filename}.png")
    plt.close()