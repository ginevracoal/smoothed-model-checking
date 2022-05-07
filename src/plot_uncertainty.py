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
import matplotlib.pyplot as plt

from settings import *
from SVI_BNNs.bnn import BNN_smMC
from EP_GPs.smMC_GPEP import smMC_GPEP
from SVI_GPs.variational_GP import GPmodel
from data_utils import get_tensor_data, normalize_columns

parser = argparse.ArgumentParser()
parser.add_argument("--svi_gp_likelihood", default='binomial', type=str, help='Choose bernoulli or binomial')
parser.add_argument("--svi_gp_variational_distribution", default='cholesky', type=str, help="Variational distribution")
parser.add_argument("--svi_gp_variational_strategy", default='default', type=str, help="Variational strategy")
parser.add_argument("--svi_gp_batch_size", default=100, type=int, help="Batch size")
parser.add_argument("--svi_gp_n_epochs", default=1000, type=int, help="Number of training iterations")
parser.add_argument("--svi_gp_lr", default=0.01, type=float, help="Learning rate")
parser.add_argument("--svi_bnn_likelihood", default='binomial', type=str, help="Choose 'bernoulli' or 'binomial'")
parser.add_argument("--svi_bnn_architecture", default='3L', type=str, help="NN architecture")
parser.add_argument("--svi_bnn_batch_size", default=100, type=int, help="Batch size")
parser.add_argument("--svi_bnn_n_epochs", default=5000, type=int, help="Number of training iterations")
parser.add_argument("--svi_bnn_lr", default=0.001, type=float, help="Learning rate")
parser.add_argument("--svi_bnn_n_hidden", default=30, type=int, help="Size of hidden layers")
parser.add_argument("--n_posterior_samples", default=1000, type=int, help="Number of samples from posterior distribution")
parser.add_argument("--plot_training_points", default=False, type=bool, help="")
parser.add_argument("--device", default="cpu", type=str, help="Choose 'cpu' or 'cuda'")
args = parser.parse_args()
print(args)

palette = sns.color_palette("magma_r", 4)
sns.set_style("darkgrid")
sns.set_palette(palette)
matplotlib.rc('font', **{'size':9, 'weight' : 'bold'})


for filepath, train_filename, val_filename, params_list, math_params_list in case_studies:

    df = pd.DataFrame()

    print(f"\n=== Eval on {val_filename} ===")

    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)

    ### Load data

    with open(os.path.join(data_path, filepath, train_filename+".pickle"), 'rb') as handle:
        train_data = pickle.load(handle)

    with open(os.path.join(data_path, filepath, val_filename+".pickle"), 'rb') as handle:
        val_data = pickle.load(handle)

    ### Validation

    x_val, y_val_bernoulli = val_data['params'], val_data['labels']
    p = y_val_bernoulli.mean(1).flatten()
    sample_variance = [((param_y-param_y.mean())**2).mean() for param_y in y_val_bernoulli]
    std = np.sqrt(sample_variance).flatten()
    n_trials_val = get_tensor_data(val_data)[3]
    errors = (1.96*std)/np.sqrt(n_trials_val)

    df = pd.concat([df, pd.DataFrame({
        "params_idx":list(range(len(val_data["params"]))),
        "uncertainty":2*errors,
        "model":"Test"
        })], ignore_index=True)

    ### Eval models on validation set
    
    print(f"\nEP GP model:")

    try:
        out_filename = f"ep_gp_{train_filename}"

        smc = smMC_GPEP()
        training_time = smc.load(filepath=os.path.join(models_path, "EP_GPs/"), filename=out_filename)

        x_train, y_train, n_samples_train, n_trials_train = smc.transform_data(train_data)
        x_val, y_val, n_samples_val, n_trials_val = smc.transform_data(val_data)
        post_mean, q1, q2, evaluation_dict = smc.eval_gp(x_train=x_train, x_val=x_val, y_val=val_data['labels'], 
            n_samples=n_samples_val, n_trials=n_trials_val)

        df = pd.concat([df, pd.DataFrame({
            "params_idx":list(range(len(val_data["params"]))),
            "uncertainty":evaluation_dict["uncertainty_area"],
            "model":"EP GP"
            })], ignore_index=True)  

    except:
        print("\nEP is unfeasible on this dataset.")

    print(f"\nSVI GP model:")

    if len(params_list)==6:
        svi_gp_n_epochs = 1000
        svi_gp_batch_size = 5000
    else:
        svi_gp_n_epochs = args.svi_gp_n_epochs
        svi_gp_batch_size = args.svi_gp_batch_size

    out_filename = f"svi_gp_{train_filename}_epochs={svi_gp_n_epochs}_lr={args.svi_gp_lr}_batch={svi_gp_batch_size}_{args.svi_gp_variational_distribution}_{args.svi_gp_variational_strategy}"

    inducing_points = normalize_columns(get_tensor_data(train_data)[0])
    model = GPmodel(inducing_points=inducing_points, variational_distribution=args.svi_gp_variational_distribution,
        variational_strategy=args.svi_gp_variational_strategy, likelihood=args.svi_gp_likelihood)
    training_time = model.load(filepath=os.path.join(models_path, "SVI_GPs/"), filename=out_filename)
        
    post_mean, q1, q2, evaluation_dict = model.evaluate(train_data=train_data, val_data=val_data, 
        n_posterior_samples=args.n_posterior_samples, device=args.device)

    df = pd.concat([df, pd.DataFrame({
        "params_idx":list(range(len(val_data["params"]))),
        "uncertainty":evaluation_dict["uncertainty_area"],
        "model":"SVI GP"
        })], ignore_index=True)

    print(f"\nSVI BNN model:")

    if len(params_list)==6:
        svi_bnn_n_epochs = 100
        svi_bnn_batch_size = 5000
    else:
        svi_bnn_n_epochs = args.svi_bnn_n_epochs
        svi_bnn_batch_size = args.svi_bnn_batch_size

    pyro.clear_param_store()

    out_filename = f"svi_bnn_{train_filename}_epochs={svi_bnn_n_epochs}_lr={args.svi_bnn_lr}_batch={svi_bnn_batch_size}_hidden={args.svi_bnn_n_hidden}_{args.svi_bnn_architecture}"
    
    bnn_smmc = BNN_smMC(model_name=filepath, list_param_names=params_list, likelihood=args.svi_bnn_likelihood,
        input_size=len(params_list), n_hidden=args.svi_bnn_n_hidden, architecture_name=args.svi_bnn_architecture)
    training_time = bnn_smmc.load(filepath=os.path.join(models_path, "SVI_BNNs/"), filename=out_filename, device=args.device)

    post_mean, q1, q2, evaluation_dict = bnn_smmc.evaluate(train_data=train_data, val_data=val_data,
        n_posterior_samples=args.n_posterior_samples, device=args.device)

    df = pd.concat([df, pd.DataFrame({
        "params_idx":list(range(len(val_data["params"]))),
        "uncertainty":evaluation_dict["uncertainty_area"],
        "model":"SVI BNN"
        })], ignore_index=True)


    if len(params_list)==1:

        ### lineplot
        fig, ax = plt.subplots(figsize=(5, 3), dpi=150, sharex=True, sharey=True)
        sns.lineplot(data=df, x="params_idx", y="uncertainty", ax=ax, hue="model", palette=palette)
        ax.set_xlabel(f"{math_params_list[0]}")
        ax.set_ylabel("Uncertainty")

        plt.tight_layout()
        plt.close()
        os.makedirs(os.path.join(plots_path), exist_ok=True)
        fig.savefig(os.path.join(plots_path, f"{val_filename}_uncertainty_lineplot.png"))

    else:

        ### boxplot
        fig, ax = plt.subplots(figsize=(4, 3), dpi=150, sharex=True, sharey=True)
        sns.boxplot(data=df, x="model", y="uncertainty", ax=ax, palette=palette, dodge=False, showfliers=False)
        ax.set_xlabel("Model")
        ax.set_ylabel("Uncertainty")

        plt.tight_layout()
        plt.close()
        os.makedirs(os.path.join(plots_path), exist_ok=True)
        fig.savefig(os.path.join(plots_path, f"{val_filename}_uncertainty_boxplot.png"))

