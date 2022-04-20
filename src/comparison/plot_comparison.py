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

sys.path.append(".")
from paths import *
from SVI_BNNs.bnn import BNN_smMC
from EP_GPs.smMC_GPEP import smMC_GPEP
from SVI_GPs.variational_GP import GPmodel
from plot_utils import plot_posterior_ax, plot_validation_ax
from data_utils import get_tensor_data, normalize_columns

parser = argparse.ArgumentParser()
parser.add_argument("--ep_gp_n_epochs", default=1000, type=int, help="Max number of training iterations")
parser.add_argument("--ep_gp_lr", default=0.01, type=float, help="Learning rate")
parser.add_argument("--svi_gp_likelihood", default='binomial', type=str, help='Choose bernoulli or binomial')
parser.add_argument("--svi_gp_variational_distribution", default='cholesky', type=str, help="Variational distribution")
parser.add_argument("--svi_gp_variational_strategy", default='default', type=str, help="Variational strategy")
parser.add_argument("--svi_gp_batch_size", default=500, type=int, help="Batch size")
parser.add_argument("--svi_gp_n_epochs", default=2000, type=int, help="Number of training iterations")
parser.add_argument("--svi_gp_lr", default=0.01, type=float, help="Learning rate")
parser.add_argument("--svi_bnn_architecture", default='2L', type=str, help="NN architecture")
parser.add_argument("--svi_bnn_batch_size", default=500, type=int, help="Batch size")
parser.add_argument("--svi_bnn_n_epochs", default=10000, type=int, help="Number of training iterations")
parser.add_argument("--svi_bnn_lr", default=0.01, type=float, help="Learning rate")
parser.add_argument("--svi_bnn_n_hidden", default=10, type=int, help="Size of hidden layers")
parser.add_argument("--n_posterior_samples", default=100, type=int, help="Number of samples from posterior distribution")
parser.add_argument("--plot_training_points", default=False, type=bool, help="")
args = parser.parse_args()


palette = sns.color_palette("magma_r", 3)
sns.set_style("darkgrid")
sns.set_palette(palette)
matplotlib.rc('font', **{'size':9, 'weight' : 'bold'})

for filepath, train_filename, val_filename, params_list, math_params_list in case_studies:

    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)

    ### Load data

    with open(os.path.join(data_path, filepath, train_filename+".pickle"), 'rb') as handle:
        train_data = pickle.load(handle)

    with open(os.path.join(data_path, filepath, val_filename+".pickle"), 'rb') as handle:
        val_data = pickle.load(handle)

    ### Set plots

    n_params = len(params_list)

    if n_params==1:
        fig, ax = plt.subplots(1, 3, figsize=(10, 3), dpi=150, sharex=True, sharey=True)

    elif n_params==2:
        fig, ax = plt.subplots(1, 4, figsize=(11, 3), dpi=150, sharex=True, sharey=True)

    ### Eval models on validation set
    
    print(f"\n=== Eval EP GP model on {val_filename} ===")

    out_filename = f"ep_gp_{train_filename}_epochs={args.ep_gp_n_epochs}_lr={args.ep_gp_lr}"

    smc = smMC_GPEP()
    smc.load(filepath=os.path.join("EP_GPs", models_path), filename=out_filename)

    x_val, y_val, n_samples_val, n_trials_val = smc.transform_data(val_data)
    post_mean, q1, q2, evaluation_dict = smc.eval_gp(x_train=x_train, x_val=x_val, y_val=val_data['labels'], 
        n_samples=n_samples_val, n_trials=n_trials_val)

    if n_params<=2:

        ax = plot_posterior_ax(ax=ax, ax_idxs=[0,1], params_list=params_list, math_params_list=math_params_list,  
            train_data=train_data, test_data=val_data, post_mean=post_mean, q1=q1, q2=q2, title='EP GP', legend='auto',
            palette=palette)

    print(f"\n=== Eval SVI GP model on {val_filename} ===")

    out_filename = f"svi_gp_{train_filename}_epochs={args.svi_gp_n_epochs}_lr={args.svi_gp_lr}_batch={args.svi_gp_batch_size}_{args.svi_gp_variational_distribution}_{args.svi_gp_variational_strategy}"

    inducing_points = normalize_columns(get_tensor_data(train_data)[0])
    model = GPmodel(inducing_points=inducing_points, variational_distribution=args.svi_gp_variational_distribution,
        variational_strategy=args.svi_gp_variational_strategy, likelihood=args.svi_gp_likelihood)
    model.load(filepath=os.path.join("SVI_GPs", models_path), filename=out_filename)
        
    post_mean, q1, q2, evaluation_dict = model.evaluate(train_data=train_data, val_data=val_data, 
        n_posterior_samples=args.n_posterior_samples)

    if n_params<=2:

        ax = plot_posterior_ax(ax=ax, ax_idxs=[1,2], params_list=params_list, math_params_list=math_params_list,  
            train_data=train_data, test_data=val_data, post_mean=post_mean, q1=q1, q2=q2, title='SVI GP', legend=None,
            palette=palette)

    print(f"\n=== Eval SVI BNN model on {val_filename} ===")

    # pyro.clear_param_store()

    out_filename = f"svi_bnn_{train_filename}_epochs={args.svi_bnn_n_epochs}_lr={args.svi_bnn_lr}_batch={args.svi_bnn_batch_size}_hidden={args.svi_bnn_n_hidden}"

    bnn_smmc = BNN_smMC(model_name=filepath, list_param_names=params_list, 
        input_size=len(params_list), n_hidden=args.svi_bnn_n_hidden, architecture_name=args.svi_bnn_architecture)

    post_mean, q1, q2, evaluation_dict = bnn_smmc.evaluate(train_data=train_data, val_data=val_data,
        n_posterior_samples=args.n_posterior_samples)

    if n_params<=2:

        ax = plot_posterior_ax(ax=ax, ax_idxs=[2,3], params_list=params_list, math_params_list=math_params_list,  
            train_data=train_data, test_data=val_data, post_mean=post_mean, q1=q1, q2=q2, title='SVI BNN', legend='auto',
            palette=palette)

        ### plot validation

        ax = plot_validation_ax(ax=ax, params_list=params_list, math_params_list=math_params_list, 
            test_data=val_data, val_data=val_data, z=1.96, palette=palette)

        ### save plot

        plt.tight_layout()
        plt.close()
        os.makedirs(os.path.join("comparison", plots_path), exist_ok=True)

        plot_filename = train_filename if val_filename is None else val_filename
        fig.savefig(os.path.join("comparison", plots_path, f"{plot_filename}.png"))