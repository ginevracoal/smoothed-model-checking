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
from GPs.variational_GP import GPmodel, MAX_N_INDUCING_PTS
from GPs.variational_GP import evaluate_GP as evaluate_var_GP
from baselineGPs.utils import evaluate_GP as evaluate_Laplace_GP
from baselineGPs.binomial_likelihood import Binomial
from GPs.binomial_likelihood import BinomialLikelihood
from data_utils import build_bernoulli_dataframe, build_binomial_dataframe, normalize_columns, Poisson_satisfaction_function


parser = argparse.ArgumentParser()
parser.add_argument("--bnn_n_epochs", default=10000, type=int, help="Number of training iterations")
parser.add_argument("--bnn_lr", default=0.01, type=float, help="Learning rate")
parser.add_argument("--bnn_n_hidden", default=10, type=int)
parser.add_argument("--bnn_architecture", default='3L', type=str)
parser.add_argument("--gp_likelihood", default='binomial', type=str, help='Choose bernoulli or binomial')
parser.add_argument("--gp_variational_distribution", default='cholesky', type=str, help="Variational distribution")
parser.add_argument("--gp_variational_strategy", default='unwhitened', type=str, help="Variational strategy")
parser.add_argument("--gp_max_n_epochs", default=1000, type=int, help="Max number of training iterations")
parser.add_argument("--gp_lr", default=0.01, type=float, help="Learning rate")
parser.add_argument("--baseline_inference", default='laplace', type=str)
parser.add_argument("--baseline_variance", default=.5, type=int, help="")
parser.add_argument("--baseline_lengthscale", default=.5, type=int, help="")
parser.add_argument("--n_posterior_samples", default=30, type=int, help="Number of samples from posterior distribution")
parser.add_argument("--plot_training_points", default=False, type=bool, help="")
parser.add_argument("--fill_ci", default=False, type=bool, help="")
args = parser.parse_args()


palette = sns.color_palette("magma_r", 3)

z=1.96

def plot_posterior_preds(filepath, ax, ax_idxs, params_list, math_params_list, x_train, y_train, x_test, 
    post_samples, post_mean, post_std, title, legend, z=1.06):

    n_params = len(params_list)

    axis = ax[ax_idxs[0]] if n_params==1 else ax[ax_idxs[1]]

    if n_params==1:

        if filepath=='Poisson':

            if args.plot_training_points:
                sns.scatterplot(x=x_train, y=y_train, ax=axis, 
                    label='Training', marker='.', color='black', alpha=0.8, legend=legend, palette=palette, linewidth=0)

            if args.fill_ci:
                sns.lineplot(x=x_test.flatten(), y=post_mean, ax=axis, label='Posterior', legend=legend, palette=palette)
                axis.fill_between(x_test.flatten(), post_mean-z*post_std, post_mean+z*post_std, alpha=0.5)
            else:
                x_test_rep = np.repeat(x_test, post_samples.shape[0])
                print(x_test_rep.shape, post_samples.shape)
                sns.lineplot(x=x_test_rep.flatten(), y=post_samples.flatten(), ax=axis, label='Posterior', 
                    palette=palette, ci=95)

            axis.set_xlabel(math_params_list[0])
            axis.set_ylabel('Satisfaction probability')
            axis.set_title(title)

        else:
            if args.plot_training_points:
                sns.scatterplot(x=x_train_binomial.flatten(), y=y_train_binomial.flatten()/n_trials_train, ax=axis, 
                    label='Training', marker='.', color='black', alpha=alpha, legend=legend, palette=palette, linewidth=0)

            if args.fill_ci:
                sns.lineplot(x=x_test.flatten(), y=post_mean, ax=axis, label='Posterior',  legend=legend, palette=palette)
                axis.fill_between(x_test.flatten(), post_mean-z*post_std, post_mean+z*post_std, alpha=0.5)
            else:
                x_test_rep = np.repeat(x_test, post_samples.shape[0])
                sns.lineplot(x=x_test_rep.flatten(), y=post_samples.flatten(), ax=axis, label='Posterior', 
                    palette=palette, ci=95)

            axis.set_xlabel(math_params_list[0])
            axis.set_ylabel('Satisfaction probability')
            axis.set_title(title)

    elif n_params==2:

        p1, p2 = params_list[0], params_list[1]

        data = pd.DataFrame({p1:x_test[:,0],p2:x_test[:,1],'posterior_preds':post_mean})
        data[p1] = data[p1].apply(lambda x: format(float(x),".2f"))
        data[p2] = data[p2].apply(lambda x: format(float(x),".2f"))
        pivot_data = data.pivot(p1, p2, "posterior_preds")
        pivot_data = pivot_data.reindex(index=data[p1].drop_duplicates(), columns=data[p2].drop_duplicates())
        sns.heatmap(pivot_data, ax=axis, label=f'{title} posterior preds')
        axis.set_title(title)
        axis.set_xlabel(math_params_list[0])
        axis.set_ylabel(math_params_list[1])

    return ax

for filepath, train_filename, val_filename, params_list, math_params_list in data_paths:
    
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)

    sns.set_style("darkgrid")
    sns.set_palette(palette)
    matplotlib.rc('font', **{'size':10, 'weight' : 'bold'})

    print(f"\n=== Eval baseline model on {val_filename} ===")

    with open(os.path.join(data_path, filepath, train_filename+".pickle"), 'rb') as handle:
        data = pickle.load(handle)

    if args.gp_likelihood=='binomial':
        x_train, y_train, n_params, n_trials_train = build_binomial_dataframe(data)
        x_train_binomial, y_train_binomial = x_train, y_train

    else:
        raise NotImplementedError

    normalized_x_train = normalize_columns(x_train).numpy()
    y_train = y_train.unsqueeze(1).numpy()

    Y_metadata = {'trials':np.full(y_train.shape, n_trials_train)}
    
    likelihood = Binomial()
    kernel = GPy.kern.RBF(input_dim=n_params, variance=args.baseline_variance, lengthscale=args.baseline_lengthscale)

    if args.baseline_inference=='laplace':
        inference = GPy.inference.latent_function_inference.Laplace()
        model = GPy.core.GP(X=normalized_x_train, Y=y_train, kernel=kernel, inference_method=inference, likelihood=likelihood, 
            Y_metadata=Y_metadata)
        
    else:
        raise NotImplementedError

    with open(os.path.join("baselineGPs", models_path, "gp_"+train_filename+".pkl"), 'rb') as file:
        model = pickle.load(file)

    file = open(os.path.join("baselineGPs", models_path,f"gp_{train_filename}_training_time.txt"),"r+")
    print(f"\nTraining time = {file.read()}")

    if filepath=='Poisson':

        x_test, post_samples, post_mean, post_std, evaluation_dict = evaluate_Laplace_GP(model=model, x_val=None, 
            y_val=None, n_trials_val=None, n_posterior_samples=args.n_posterior_samples, n_params=n_params)

    else: 

        with open(os.path.join(data_path, filepath, val_filename+".pickle"), 'rb') as handle:
            val_data = pickle.load(handle)
        
        x_val, y_val, n_params, n_trials_val = build_binomial_dataframe(val_data)

        x_test, post_samples, post_mean, post_std, evaluation_dict = evaluate_Laplace_GP(model=model, x_val=x_val, 
            y_val=y_val, n_trials_val=n_trials_val, n_posterior_samples=args.n_posterior_samples, n_params=n_params)

    if n_params==1:
        fig, ax = plt.subplots(1, 3, figsize=(10, 3), dpi=150, sharex=True, sharey=True)

    elif n_params==2:
        fig, ax = plt.subplots(1, 4, figsize=(11, 3), dpi=150, sharex=True, sharey=True)

    ### plot Laplace

    ax = plot_posterior_preds(filepath=filepath, ax=ax, ax_idxs=[0,1], params_list=params_list, 
        math_params_list=math_params_list, 
        x_train=x_train_binomial.flatten(), y_train=y_train_binomial.flatten()/n_trials_train, x_test=x_test, 
        post_samples=post_samples, post_mean=post_mean, post_std=post_std, title='Laplace GP', legend=None)

    print(f"\n=== Eval GP model on {val_filename} ===")

    full_path = os.path.join(data_path, filepath, train_filename+".pickle")
    with open(full_path, 'rb') as handle:
        print(f"\nLoading {full_path}")
        data = pickle.load(handle)

    if args.gp_likelihood=='binomial':
        x_train, y_train, n_params, n_trials_train = build_binomial_dataframe(data)
        x_train_binomial, y_train_binomial = x_train, y_train
        likelihood = BinomialLikelihood()
    else:
        raise NotImplementedError

    normalized_x_train = normalize_columns(x_train) 
    inducing_points = normalize_columns(x_train_binomial)
    
    gp_n_epochs = 100 if len(inducing_points)>MAX_N_INDUCING_PTS else args.gp_max_n_epochs
    out_filename = f"{args.gp_likelihood}_{train_filename}_epochs={gp_n_epochs}_lr={args.gp_lr}"

    model = GPmodel(inducing_points=inducing_points, variational_distribution=args.gp_variational_distribution,
        variational_strategy=args.gp_variational_strategy)

    state_dict = torch.load(os.path.join(os.path.join("GPs", models_path), "gp_state_"+out_filename+".pth"))
    model.load_state_dict(state_dict)

    file = open(os.path.join("GPs", models_path,f"gp_{out_filename}_training_time.txt"),"r+")
    print(f"\nTraining time = {file.read()}")

    if filepath=='Poisson':

        x_test, post_samples, post_mean, post_std, evaluation_dict = evaluate_var_GP(model=model, likelihood=likelihood,
            n_posterior_samples=args.n_posterior_samples)

    else: 

        with open(os.path.join(data_path, filepath, val_filename+".pickle"), 'rb') as handle:
            val_data = pickle.load(handle)
        
        x_val, y_val, n_params, n_trials_val = build_binomial_dataframe(val_data)

        x_test, post_samples, post_mean, post_std, evaluation_dict = evaluate_var_GP(model=model, likelihood=likelihood, 
            x_val=x_val, y_val=y_val, n_trials_val=n_trials_val, n_posterior_samples=args.n_posterior_samples)

    ax = plot_posterior_preds(filepath=filepath, ax=ax, ax_idxs=[1,2], params_list=params_list, 
        math_params_list=math_params_list, 
        x_train=x_train_binomial.flatten(), y_train=y_train_binomial.flatten()/n_trials_train, x_test=x_test, 
        post_samples=post_samples, post_mean=post_mean, post_std=post_std, title='GP', legend=None)

    print(f"\n=== Eval BNN model on {val_filename} ===")

    df_file_train = os.path.join(os.path.join(data_path, filepath, train_filename+".pickle"))
    df_file_val = os.path.join(os.path.join(data_path, filepath, val_filename+".pickle")) if val_filename else df_file_train

    pyro.clear_param_store()
    n_test_points = 100 if filepath=='Poisson' else args.n_posterior_samples
    bnn_smmc = BNN_smMC(model_name=filepath, list_param_names=params_list, train_set=df_file_train, val_set=df_file_val, 
        input_size=len(params_list), n_hidden=args.bnn_n_hidden, n_test_points=n_test_points,
        architecture_name=args.bnn_architecture)

    x_test, post_samples, post_mean, post_std, evaluation_dict = bnn_smmc.run(n_epochs=args.bnn_n_epochs, lr=args.bnn_lr, 
        train_flag=False, n_posterior_samples=args.n_posterior_samples)

    ax = plot_posterior_preds(filepath=filepath, ax=ax, ax_idxs=[2,3], params_list=params_list, 
        math_params_list=math_params_list, 
        x_train=x_train_binomial.flatten(), y_train=y_train_binomial.flatten()/n_trials_train, x_test=x_test, 
        post_samples=post_samples, post_mean=post_mean, post_std=post_std, title='BNN', legend='auto')


    ### plot validation

    if n_params==1:

        if filepath=='Poisson':
            for idx in range(len(ax)):
                legend = 'auto' if idx==len(ax)-1 else None
                sns.lineplot(x=x_test.flatten(), y=Poisson_satisfaction_function(x_test).flatten(), ax=ax[idx], 
                    label='true satisfaction',  legend=legend, palette=palette)

        else:
            for idx in range(len(ax)):
                legend = 'auto' if idx==len(ax)-1 else None
                sns.scatterplot(x=x_val.flatten(), y=y_val.flatten()/n_trials_val, ax=ax[idx], label='Validation', 
                    legend=legend, palette=palette, linewidth=0)

    elif n_params==2:

        axis = ax[0]
        p1, p2 = params_list[0], params_list[1]

        data = pd.DataFrame({p1:x_val[:,0],p2:x_val[:,1],'val_counts':y_val.flatten()/n_trials_val})
        data[p1] = data[p1].apply(lambda x: format(float(x),".2f"))
        data[p2] = data[p2].apply(lambda x: format(float(x),".2f"))
        pivot_data = data.pivot(p1, p2, "val_counts")
        pivot_data = pivot_data.reindex(index=data[p1].drop_duplicates(), columns=data[p2].drop_duplicates())
        sns.heatmap(pivot_data, ax=axis, label='Validation')
        axis.set_title("Validation set")
        axis.set_xlabel(math_params_list[0])
        axis.set_ylabel(math_params_list[1])

    ### save plots 

    if n_params<=2:
        plt.tight_layout()
        plt.close()
        os.makedirs(os.path.join("comparison", plots_path), exist_ok=True)

        plot_filename = train_filename if val_filename is None else val_filename
        fig.savefig(os.path.join("comparison", plots_path, f"{plot_filename}.png"))