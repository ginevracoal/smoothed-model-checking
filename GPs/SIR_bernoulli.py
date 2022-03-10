import os
import sys
import torch
import gpytorch
import numpy as np
import pandas as pd
import seaborn as sns
import pickle5 as pickle
import matplotlib.pyplot as plt
import torch.utils.data as data_utils

from variational_GP import GPmodel, train_GP 

def build_dataframe(data):
    params = torch.tensor(data['params'], dtype=torch.float32)
    labels = torch.tensor(data['labels'], dtype=torch.int)

    n_params = params.shape[1]
    n_trials = labels.shape[1]

    success_counts = [len(row[row==1.]) for row in labels]
    success_counts = torch.tensor(success_counts, dtype=torch.float32)

    print("\nparams shape =", params.shape)
    print("labels shape =", labels.shape)
    print("n. trials =", n_trials)
    # print("\nParams True label counts:\n", success_counts)

    return params, success_counts, n_params, n_trials

def build_bernoulli_dataframe(data):
    params = torch.tensor(data['params'], dtype=torch.float32)
    params_observations = data['labels'].shape[1]
    params = np.repeat(params, params_observations, axis=1).flatten()

    n_params = data['params'].shape[1]
    n_trials=1

    labels = torch.tensor(data['labels'], dtype=torch.int).flatten()
    
    success_counts = [len(row[row==1.]) for row in labels]
    success_counts = torch.tensor(success_counts, dtype=torch.float32)

    print("\nparams shape =", params.shape)
    print("labels shape =", labels.shape)
    print("n. trials =", n_trials)
    # print("\nParams True label counts:\n", success_counts)

    return params, success_counts, n_params, n_trials

for train_filename, val_filename in [
    ["SIR_DS_200samples_10obs_Beta", "SIR_DS_20samples_5000obs_Beta"],
    # ["SIR_DS_200samples_10obs_Gamma", "SIR_DS_20samples_5000obs_Gamma"],
    # ["SIR_DS_256samples_5000obs_BetaGamma", "SIR_DS_256samples_10obs_BetaGamma"]
    ]:

    print(f"\n=== Training {train_filename} ===")

    with open(f"../Data/SIR/{train_filename}.pickle", 'rb') as handle:
        data = pickle.load(handle)
    x_train, y_train, n_params, n_trials = build_bernoulli_dataframe(data)

    model = GPmodel(inducing_points=x_train)
    likelihood = gpytorch.likelihoods.BernoulliLikelihood()

    model = train_GP(model=model, likelihood=likelihood, x_train=x_train, y_train=y_train, num_epochs=100)
    os.makedirs(os.path.dirname('models/'), exist_ok=True)
    torch.save(model.state_dict(), f'models/gp_state_{train_filename}.pth')

    print(f"\n=== Validation {val_filename} ===")

    model.eval()    
    likelihood.eval()

    with open(f"../Data/SIR/{val_filename}.pickle", 'rb') as handle:
        data = pickle.load(handle)
    x_val, y_val, n_params, n_trials = build_dataframe(data)

    model = GPmodel(inducing_points=x_train)
    state_dict = torch.load(f'models/gp_state_{train_filename}.pth')
    model.load_state_dict(state_dict)

    likelihood = gpytorch.likelihoods.BernoulliLikelihood()

    with torch.no_grad():

        x_test = []
        for col_idx in range(n_params):
            single_param_values = x_val[:,col_idx]
            x_test.append(torch.linspace(single_param_values.min(), single_param_values.max(), 10))
        x_test = torch.stack(x_test, dim=1)

        multinorm_loc = model(x_test).mean

        # observed_pred = likelihood(model(x_test), n_trials=n_trials) 
        # pred_samples = observed_pred.sample(sample_shape=torch.Size((1000,)))

        # print("\npred_samples.shape =", pred_samples.shape, "= (n. binomial samples, n. test params)")
        # pred_probs = pred_samples.mean(dim=0)/n_trials
        # pred_mean = np.mean(pred_probs, axis=0)
        # pred_variance = np.var(pred_probs, axis=0)

    path='plots/SIR/'
    os.makedirs(os.path.dirname(path), exist_ok=True)

    fig, ax = plt.subplots(1, n_params, figsize=(6*n_params, 5))

    for col_idx in range(n_params):
        single_param_x_val = x_val[:,col_idx]
        single_param_x_test = x_test[:,col_idx]

        axis = ax if n_params==1 else ax[col_idx]
        sns.scatterplot(x=single_param_x_val, y=y_val/n_trials, ax=axis, label='validation pts')
        sns.lineplot(x=single_param_x_test, y=multinorm_loc, ax=axis, label='avg satisfaction')

        # sns.lineplot(x=single_param_x_test, y=pred_mean, ax=axis, label='pred satisfaction')
        # axis.fill_between(single_param_x_test.numpy(), pred_mean-pred_variance, pred_mean+pred_variance, alpha=0.5)

    fig.savefig(path+f"lineplot_{val_filename}.png")
    plt.close()


