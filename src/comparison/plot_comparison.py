import os
import sys
import pyro
import torch
import random
import argparse
import matplotlib
import numpy as np
import pandas as pd
import seaborn as sns
import pickle5 as pickle
from itertools import product
import matplotlib.pyplot as plt

sys.path.append(".")
from paths import *
from BNNs.bnn import BNN_smMC
from GPs.variational_GP import GPmodel, evaluate_GP
from GPs.binomial_likelihood import BinomialLikelihood
from GPs.utils import build_bernoulli_dataframe, build_binomial_dataframe, normalize_columns, Poisson_satisfaction_function

random.seed(0)
np.random.seed(0)
torch.manual_seed(0)

parser = argparse.ArgumentParser()
parser.add_argument("--bnn_n_epochs", default=10000, type=int, help="Number of training iterations")
parser.add_argument("--bnn_lr", default=0.01, type=float, help="Learning rate")
parser.add_argument("--bnn_identifier", default=1, type=int)
parser.add_argument("--bnn_n_hidden", default=10, type=int)
parser.add_argument("--bnn_architecture", default='2L', type=str)
parser.add_argument("--gp_likelihood", default='binomial', type=str, help='Choose bernoulli or binomial')
parser.add_argument("--gp_variational_distribution", default='cholesky', type=str, help="Variational distribution")
parser.add_argument("--gp_variational_strategy", default='unwhitened', type=str, help="Variational strategy")
parser.add_argument("--gp_n_epochs", default=1000, type=int, help="Number of training iterations")
parser.add_argument("--gp_lr", default=0.01, type=float, help="Learning rate")
parser.add_argument("--gp_n_posterior_samples", default=10, type=int, help="Number of samples from posterior distribution")
args = parser.parse_args()


z=1.96
palette = sns.color_palette("magma_r", 3)


for filepath, train_filename, val_filename, params_list in data_paths:

    sns.set_style("darkgrid")
    sns.set_palette(palette)
    matplotlib.rc('font', **{'size':10, 'weight' : 'bold'})

    out_filename = f"{args.gp_likelihood}_{train_filename}_epochs={args.gp_n_epochs}_lr={args.gp_lr}"

    print(f"\n=== Loading GP model {out_filename} ===")

    with open(os.path.join(data_path, filepath, train_filename+".pickle"), 'rb') as handle:
        data = pickle.load(handle)

    if args.gp_likelihood=='binomial':
        x_train, y_train, n_params, n_trials_train = build_binomial_dataframe(data)
        x_train_binomial, y_train_binomial = x_train, y_train
        likelihood = BinomialLikelihood()
    else:
        raise NotImplementedError

    normalized_x_train = normalize_columns(x_train) 
    inducing_points = normalize_columns(x_train_binomial)

    model = GPmodel(inducing_points=inducing_points, variational_distribution=args.gp_variational_distribution,
        variational_strategy=args.gp_variational_strategy)

    state_dict = torch.load(os.path.join(os.path.join("GPs", models_path), "gp_state_"+out_filename+".pth"))
    model.load_state_dict(state_dict)

    file = open(os.path.join("GPs", models_path,f"gp_{out_filename}_training_time.txt"),"r+")
    print(f"\nTraining time = {file.read()}")

    if filepath=='Poisson':

        x_test, post_mean, post_std, evaluation_dict = evaluate_GP(model=model, likelihood=likelihood,
            n_posterior_samples=args.gp_n_posterior_samples, n_params=n_params)

    else: 

        with open(os.path.join(data_path, filepath, val_filename+".pickle"), 'rb') as handle:
            val_data = pickle.load(handle)
        
        x_val, y_val, n_params, n_trials_val = build_binomial_dataframe(val_data)

        x_test, post_mean, post_std, evaluation_dict = evaluate_GP(model=model, likelihood=likelihood, x_val=x_val, y_val=y_val, 
            n_trials_val=n_trials_val, n_posterior_samples=args.gp_n_posterior_samples, n_params=n_params)


    if n_params==1:

        fig, ax = plt.subplots(1, 2, figsize=(10, 4), dpi=100, sharex=True, sharey=True)

        if filepath=='Poisson':

            sns.lineplot(x=x_test.flatten(), y=post_mean, ax=ax[0], label='posterior', legend=None, palette=palette)
            ax[0].fill_between(x_test.flatten(), post_mean-z*post_std, post_mean+z*post_std, alpha=0.5)

            sns.lineplot(x=x_test.flatten(), y=Poisson_satisfaction_function(x_test).flatten(), ax=ax[0], 
                label='true satisfaction',  legend=None, palette=palette)
            sns.scatterplot(x=x_train_binomial.flatten(), y=y_train_binomial.flatten()/n_trials_train, ax=ax[0], 
                label='training points', marker='.', color='black',  legend=None, palette=palette)
            ax[0].set_xlabel(params_list[0])
            ax[0].set_ylabel('Satisfaction probability')
            ax[0].set_title('GP')

        else:
            sns.scatterplot(x=x_train_binomial.flatten(), y=y_train_binomial.flatten()/n_trials_train, ax=ax[0], 
                label='training points', marker='.', color='black',  legend=None, palette=palette)
            sns.lineplot(x=x_test.flatten(), y=post_mean, ax=ax[0], label='posterior',  legend=None, palette=palette)
            ax[0].fill_between(x_test.flatten(), post_mean-z*post_std, post_mean+z*post_std, alpha=0.5)
            sns.scatterplot(x=x_val.flatten(), y=y_val.flatten()/n_trials_val, ax=ax[0], label='validation pts', 
                legend=None, palette=palette)
            ax[0].set_xlabel(params_list[0])
            ax[0].set_ylabel('Satisfaction probability')
            ax[0].set_title('GP')

    elif n_params==2:

        fig, ax = plt.subplots(1, 3, figsize=(13, 4), dpi=100)

        p1, p2 = params_list[0], params_list[1]

        print("gp", post_mean.shape)

        data = pd.DataFrame({p1:x_val[:,0],p2:x_val[:,1],'val_counts':y_val.flatten()/n_trials_val})
        data[p1] = data[p1].apply(lambda x: format(float(x),".4f"))
        data[p2] = data[p2].apply(lambda x: format(float(x),".4f"))
        data.sort_index(level=0, ascending=True, inplace=True)
        data = data.pivot(p1, p2, "val_counts")
        sns.heatmap(data, ax=ax[0], label='validation pts')
        ax[0].set_title("Validation set")

        data = pd.DataFrame({p1:x_test[:,0],p2:x_test[:,1],'posterior_preds':post_mean})
        data[p1] = data[p1].apply(lambda x: format(float(x),".4f"))
        data[p2] = data[p2].apply(lambda x: format(float(x),".4f"))
        data.sort_index(level=0, ascending=True, inplace=True)
        data = data.pivot(p1, p2, "posterior_preds")
        sns.heatmap(data, ax=ax[1], label='GP posterior preds')
        ax[1].set_title("GP")

    print("\n=== Loading BNN model ===")

    df_file_train = os.path.join(os.path.join(data_path, filepath, train_filename+".pickle"))
    df_file_val = os.path.join(os.path.join(data_path, filepath, val_filename+".pickle")) if val_filename else df_file_train

    pyro.clear_param_store()
    n_test_points = 100 if filepath=='Poisson' else len(x_val)
    bnn_smmc = BNN_smMC(model_name=filepath, list_param_names=params_list, train_set=df_file_train, val_set=df_file_val, 
        input_size=len(params_list), n_hidden=args.bnn_n_hidden, n_test_points=n_test_points,
        architecture_name=args.bnn_architecture)

    x_test, post_mean, post_std, evaluation_dict = bnn_smmc.run(n_epochs=args.bnn_n_epochs, lr=args.bnn_lr, 
        identifier=args.bnn_identifier, train_flag=False)

    if n_params==1:

        post_mean, post_std = post_mean.flatten(), post_std.flatten()

        if bnn_smmc.model_name == "Poisson":

            sns.lineplot(x=x_test.flatten(), y=post_mean, ax=ax[1], label='posterior', palette=palette)
            ax[1].fill_between(x_test.flatten(), post_mean-z*post_std, post_mean+z*post_std, alpha=0.5)

            sns.lineplot(x=x_test.flatten(), y=Poisson_satisfaction_function(x_test).flatten(), ax=ax[1], 
                label='true satisfaction', palette=palette)
            sns.scatterplot(x=x_train_binomial.flatten(), y=y_train_binomial.flatten()/n_trials_train, ax=ax[1], 
                label='training points', marker='.', color='black', palette=palette)
            ax[1].set_xlabel(params_list[0])
            ax[1].set_title('BNN')

        else: 
            sns.scatterplot(x=bnn_smmc.X_train.flatten(), y=bnn_smmc.T_train.flatten()/bnn_smmc.M_train, ax=ax[1], 
                label='training points', marker='.', color='black', palette=palette)
            sns.lineplot(x=x_test.flatten(), y=post_mean, ax=ax[1], label='posterior', palette=palette)
            ax[1].fill_between(x_test.flatten(), post_mean-z*post_std, post_mean+z*post_std, alpha=0.5)
            sns.scatterplot(x=bnn_smmc.X_val.flatten(), y=bnn_smmc.T_val.flatten()/bnn_smmc.M_val, 
                ax=ax[1], label='validation pts', palette=palette)
            ax[1].set_xlabel(params_list[0])
            ax[1].set_title('BNN')

    elif n_params==2:

        data = pd.DataFrame({p1:x_test[:,0],p2:x_test[:,1],'posterior_preds':post_mean.flatten()})
        data[p1] = data[p1].apply(lambda x: format(float(x),".4f"))
        data[p2] = data[p2].apply(lambda x: format(float(x),".4f"))
        data.sort_index(level=0, ascending=True, inplace=True)
        data = data.pivot(p1, p2, "posterior_preds")
        sns.heatmap(data, ax=ax[2])
        ax[2].set_title("BNN")

        ax[2].set_xlabel(params_list[0])
        ax[2].set_ylabel(params_list[1])


    if fig:
        plt.tight_layout()
        plt.close()
        os.makedirs(os.path.join("comparison", plots_path), exist_ok=True)
        fig.savefig(os.path.join("comparison", plots_path, f"{train_filename}.png"))