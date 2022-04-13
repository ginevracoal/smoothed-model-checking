import os
import sys
import GPy
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
from VIGPs.variational_GP import GPmodel
from baselineGPs.binomial_likelihood import Binomial
from plot_utils import plot_posterior_ax, plot_validation_ax
from baselineGPs.utils import evaluate_GP as evaluate_Laplace_GP
from data_utils import get_bernoulli_data, get_binomial_data, get_tensor_data, normalize_columns, Poisson_satisfaction_function

parser = argparse.ArgumentParser()
parser.add_argument("--bnn_n_epochs", default=10000, type=int, help="Number of training iterations")
parser.add_argument("--bnn_lr", default=0.01, type=float, help="Learning rate")
parser.add_argument("--bnn_n_hidden", default=10, type=int)
parser.add_argument("--bnn_architecture", default='3L', type=str)
parser.add_argument("--gp_likelihood", default='binomial', type=str, help='Choose bernoulli or binomial')
parser.add_argument("--gp_variational_distribution", default='cholesky', type=str, help="Variational distribution")
parser.add_argument("--gp_variational_strategy", default='unwhitened', type=str, help="Variational strategy")
parser.add_argument("--gp_n_epochs", default=1000, type=int, help="Max number of training iterations")
parser.add_argument("--gp_lr", default=0.01, type=float, help="Learning rate")
parser.add_argument("--baseline_inference", default='laplace', type=str)
parser.add_argument("--baseline_variance", default=.5, type=int, help="")
parser.add_argument("--baseline_lengthscale", default=.5, type=int, help="")
parser.add_argument("--n_posterior_samples", default=30, type=int, help="Number of samples from posterior distribution")
parser.add_argument("--plot_training_points", default=False, type=bool, help="")
args = parser.parse_args()


palette = sns.color_palette("magma_r", 3)
sns.set_style("darkgrid")
sns.set_palette(palette)
matplotlib.rc('font', **{'size':9, 'weight' : 'bold'})

for filepath, train_filename, val_filename, params_list, math_params_list in data_paths:

    n_params = len(params_list)

    if n_params==1:
        fig, ax = plt.subplots(1, 3, figsize=(10, 3), dpi=150, sharex=True, sharey=True)

    elif n_params==2:
        fig, ax = plt.subplots(1, 4, figsize=(11, 3), dpi=150, sharex=True, sharey=True)

    if n_params<=2:
    
        random.seed(0)
        np.random.seed(0)
        torch.manual_seed(0)

        print(f"\n=== Eval baseline model on {val_filename} ===")

        with open(os.path.join(data_path, filepath, train_filename+".pickle"), 'rb') as handle:
            train_data = pickle.load(handle)

        if args.gp_likelihood=='binomial':
            x_train, y_train, n_samples, n_trials_train = get_binomial_data(train_data)
            likelihood = Binomial()
            x_train = normalize_columns(x_train).numpy()
            y_train = y_train.unsqueeze(1).numpy()

        else:
            raise NotImplementedError

        Y_metadata = {'trials':np.full(y_train.shape, n_trials_train)}
        
        likelihood = Binomial()
        kernel = GPy.kern.RBF(input_dim=n_params, variance=args.baseline_variance, lengthscale=args.baseline_lengthscale)

        if args.baseline_inference=='laplace':
            inference = GPy.inference.latent_function_inference.Laplace()
            model = GPy.core.GP(X=x_train, Y=y_train, kernel=kernel, inference_method=inference, likelihood=likelihood, 
                Y_metadata=Y_metadata)
            
        else:
            raise NotImplementedError

        with open(os.path.join("baselineGPs", models_path, "gp_"+train_filename+".pkl"), 'rb') as file:
            model = pickle.load(file)

        file = open(os.path.join("baselineGPs", models_path,f"gp_{train_filename}_training_time.txt"),"r+")
        print(f"\nTraining time = {file.read()}")

        if filepath=='Poisson':
            raise NotImplementedError
            # x_test, post_samples, post_mean, post_std, q1,q2, evaluation_dict = evaluate_Laplace_GP(model=model, x_val=None, 
            #     y_val=None, n_trials_val=None, n_posterior_samples=args.n_posterior_samples, n_params=n_params)

        else: 

            with open(os.path.join(data_path, filepath, val_filename+".pickle"), 'rb') as handle:
                val_data = pickle.load(handle)
            
            post_mean, q1, q2, evaluation_dict = evaluate_Laplace_GP(model=model, val_data=val_data,
                n_samples=n_samples, n_posterior_samples=args.n_posterior_samples)

        ax = plot_posterior_ax(ax=ax, ax_idxs=[0,1], params_list=params_list, math_params_list=math_params_list,  
            train_data=train_data, test_data=val_data, post_mean=post_mean, q1=q1, q2=q2, title='Laplace GP', legend='auto',
            palette=palette)


        print(f"\n=== Eval GP model on {val_filename} ===")

        full_path = os.path.join(data_path, filepath, train_filename+".pickle")
        with open(full_path, 'rb') as handle:
            print(f"\nLoading {full_path}")
            train_data = pickle.load(handle)

        inducing_points = normalize_columns(get_tensor_data(train_data)[0])
        
        gp_n_epochs = args.gp_n_epochs #if len(inducing_points)>1000 else args.max_n_epochs
        out_filename = f"{args.gp_likelihood}_{train_filename}_epochs={gp_n_epochs}_lr={args.gp_lr}"

        model = GPmodel(inducing_points=inducing_points, variational_distribution=args.gp_variational_distribution,
            variational_strategy=args.gp_variational_strategy, likelihood=args.gp_likelihood)

        model.load(filepath=os.path.join("VIGPs", models_path), filename=out_filename)

        if filepath=='Poisson':
            raise NotImplementedError

        else: 
            with open(os.path.join(data_path, filepath, val_filename+".pickle"), 'rb') as handle:
                val_data = pickle.load(handle)
            
            post_mean, q1, q2, evaluation_dict = model.eval_GP(val_data=val_data, n_posterior_samples=args.n_posterior_samples)

        ax = plot_posterior_ax(ax=ax, ax_idxs=[1,2], params_list=params_list, math_params_list=math_params_list,  
            train_data=train_data, test_data=val_data, post_mean=post_mean, q1=q1, q2=q2, title='GP', legend=None,
            palette=palette)

        print(f"\n=== Eval BNN model on {val_filename} ===")

        df_file_train = os.path.join(os.path.join(data_path, filepath, train_filename+".pickle"))
        df_file_val = os.path.join(os.path.join(data_path, filepath, val_filename+".pickle")) if val_filename else df_file_train

        pyro.clear_param_store()
        n_test_points = 100 if filepath=='Poisson' else len(val_data['params'])
        bnn_smmc = BNN_smMC(model_name=filepath, list_param_names=params_list, train_set=df_file_train, val_set=df_file_val, 
            input_size=len(params_list), n_hidden=args.bnn_n_hidden, n_test_points=n_test_points,
            architecture_name=args.bnn_architecture)

        x_test, post_samples, post_mean, q1, q2, evaluation_dict = bnn_smmc.run(n_epochs=args.bnn_n_epochs, lr=args.bnn_lr, 
            y_val=val_data['labels'], train_flag=False, n_posterior_samples=args.n_posterior_samples)

        ax = plot_posterior_ax(ax=ax, ax_idxs=[2,3], params_list=params_list, math_params_list=math_params_list,  
            train_data=train_data, test_data=val_data, post_mean=post_mean, q1=q1, q2=q2, title='BNN', legend='auto',
            palette=palette)

        ### plot validation

        ax = plot_validation_ax(ax=ax, params_list=params_list, math_params_list=math_params_list, 
            test_data=val_data, val_data=val_data, z=1.96, palette=palette)

        ### save plots 

        plt.tight_layout()
        plt.close()
        os.makedirs(os.path.join("comparison", plots_path), exist_ok=True)

        plot_filename = train_filename if val_filename is None else val_filename
        fig.savefig(os.path.join("comparison", plots_path, f"{plot_filename}.png"))