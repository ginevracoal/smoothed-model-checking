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

import GPy
import climin

from data_utils import build_binomial_dataframe

parser = argparse.ArgumentParser()
parser.add_argument("--variational_distribution", default='cholesky', type=str, help="Variational distribution")
parser.add_argument("--variational_strategy", default='default', type=str, help="Variational strategy")
parser.add_argument("--train", default=True, type=eval, help="If True train the model else load it")
parser.add_argument("--n_epochs", default=1000, type=int, help="Number of training iterations")
parser.add_argument("--lr", default=0.01, type=float, help="Learning rate")
parser.add_argument("--n_test_points", default=100, type=int, help="Number of test params")
parser.add_argument("--n_posterior_samples", default=1000, type=int, help="Number of samples from posterior distribution")
args = parser.parse_args()



for train_filename, val_filename in [
    ["SIR_DS_200samples_10obs_Beta", "SIR_DS_20samples_5000obs_Beta"],
    # ["SIR_DS_200samples_10obs_Gamma", "SIR_DS_20samples_5000obs_Gamma"],
    # ["SIR_DS_256samples_5000obs_BetaGamma", "SIR_DS_256samples_10obs_BetaGamma"]
    ]:

    print(f"\n=== Training {train_filename} ===")

    out_filename = f"binomial_{train_filename}_epochs={args.n_epochs}_lr={args.lr}"

    with open(f"../Data/SIR/{train_filename}.pickle", 'rb') as handle:
        data = pickle.load(handle)
    x_train, y_train, n_params, n_trials_train = build_binomial_dataframe(data)

    x_train = x_train.detach().numpy()

    # this library only takes y_train with 2 cols, so I'm duplicating the first col
    y_train_duplicated = y_train.repeat(2,1).t().detach().numpy()

    kernel = GPy.kern.RBF(1) 
    likelihood = GPy.likelihoods.Binomial()
    # inducing_points = x_train 

    inducing_points = []
    for col_idx in range(n_params):
        inducing_points.append(np.random.uniform(low=x_train[:,col_idx].min(), high=x_train[:,col_idx].max(), size=(100,1)))
    inducing_points = np.hstack(inducing_points)

    Y_metadata = {'trials':n_trials_train}

    model = GPy.core.SVGP(x_train, y_train_duplicated, inducing_points, 
        kernel=kernel, likelihood=likelihood, batchsize=len(x_train), Y_metadata=Y_metadata)

    if args.train:

        opt = climin.Adam(model.optimizer_array, model.stochastic_grad, step_rate=0.01, momentum=0.9)

        def callback(i):
            # print(f"Epoch {i}/{args.n_epochs} - Loss: {str(model.log_likelihood())}")

            print(str(model.log_likelihood()))
            if i['n_iter'] > args.n_epochs:
                return True
            return False
        info = opt.minimize_until(callback)

        os.makedirs(os.path.dirname('models/'), exist_ok=True)
        np.save(f'models/gp_state_{out_filename}.npy', model.param_array)

    print(f"\n=== Validation {val_filename} ===")

    model.initialize_parameter() 
    model[:] = np.load(f'models/gp_state_{out_filename}.npy') 
    model.update_model(True) 

    with open(f"../Data/SIR/{val_filename}.pickle", 'rb') as handle:
        data = pickle.load(handle)
    x_val, y_val, n_params, n_trials_val = build_binomial_dataframe(data)

    x_test = []
    for col_idx in range(n_params):
        single_param_values = x_val[:,col_idx]
        x_test.append(torch.linspace(single_param_values.min(), single_param_values.max(), 100))
    x_test = torch.stack(x_test, dim=1).detach().numpy()

    Y_metadata = {'trials':n_trials_val}
    pred_samples = model.posterior_samples(x_test, size=args.n_posterior_samples, Y_metadata=Y_metadata)[:,0,:].transpose()

    # rbf = model.parameters[1]
    # binomial = model.parameters[2]
    # SVGP_variational_mean = model.parameters[4]

    # rbf_lengthscale = model.rbf.lengthscale.values[0]
    # rbf_variance = np.sqrt(model.rbf.variance.values[0])


    print("\npred_samples.shape =", pred_samples.shape, "= (n. binomial samples, n. test params)")
    pred_probs = pred_samples/n_trials_val
    mu = np.mean(pred_probs, axis=0)
    sigma = np.var(pred_probs, axis=0)

    path='plots/SIR/'
    os.makedirs(os.path.dirname(path), exist_ok=True)

    fig, ax = plt.subplots(1, n_params, figsize=(6*n_params, 5))

    # for col_idx in range(n_params):
    #     single_param_x_train = x_train[:, col_idx]
    #     single_param_x_val = x_val[:,col_idx]
    #     single_param_x_test = x_test[:,col_idx]

    #     axis = ax if n_params==1 else ax[col_idx]
    #     sns.scatterplot(x=single_param_x_val, y=y_val/n_trials_val, ax=axis, label='validation pts')

    #     sns.lineplot(x=single_param_x_test, y=pred_mean, ax=axis, label='pred satisfaction')
    #     axis.fill_between(single_param_x_test, pred_mean-pred_variance, pred_mean+pred_variance, alpha=0.5)
    #     sns.scatterplot(x=single_param_x_train, y=y_train/n_trials_train, ax=axis, label='training pts', 
    #         marker='.', color='black')

    if n_params==1:

        fig, ax = plt.subplots(1, 1, figsize=(6*n_params, 5))

        sns.scatterplot(x=x_val.flatten(), y=y_val.flatten()/n_trials_val, ax=ax, label='validation pts')
        sns.scatterplot(x=x_train.flatten(), y=y_train.flatten()/n_trials_train, ax=ax, 
            label='training points', marker='.', color='black')

        sns.lineplot(x=x_test.flatten(), y=mu, ax=ax, label='posterior')
        ax.fill_between(x_test.flatten(), mu-sigma, mu+sigma, alpha=0.5)

    else:
        raise NotImplementedError


    
    fig.savefig(path+f"gpy_{out_filename}.png")
    plt.close()


