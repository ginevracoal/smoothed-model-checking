import os
import sys
import pyro
import torch
import random
import argparse
import numpy as np
import pickle5 as pickle

sys.path.append(".")
from paths import *
from BNNs.bnn import BNN_smMC
from GPs.variational_GP import GPmodel, evaluate_GP
from GPs.binomial_likelihood import BinomialLikelihood
from GPs.utils import build_bernoulli_dataframe, build_binomial_dataframe, normalize_columns

random.seed(0)
np.random.seed(0)
torch.manual_seed(0)

parser = argparse.ArgumentParser()
parser.add_argument("--bnn_n_epochs", default=10000, type=int, help="Number of training iterations")
parser.add_argument("--bnn_lr", default=0.01, type=float, help="Learning rate")
parser.add_argument("--bnn_identifier", default=1, type=int)
parser.add_argument("--bnn_n_hidden", default=10, type=int)
parser.add_argument("--gp_likelihood", default='binomial', type=str, help='Choose bernoulli or binomial')
parser.add_argument("--gp_variational_distribution", default='cholesky', type=str, help="Variational distribution")
parser.add_argument("--gp_variational_strategy", default='unwhitened', type=str, help="Variational strategy")
parser.add_argument("--gp_n_epochs", default=1000, type=int, help="Number of training iterations")
parser.add_argument("--gp_lr", default=0.01, type=float, help="Learning rate")
parser.add_argument("--gp_n_posterior_samples", default=10, type=int, help="Number of samples from posterior distribution")
args = parser.parse_args()


os.makedirs(os.path.join("comparison", plots_path), exist_ok=True)



for filepath, train_filename, val_filename, params_list in data_paths:

    print("\n=== Loading BNN model ===")

    df_file_train = os.path.join(os.path.join(data_path, filepath, train_filename+".pickle"))
    df_file_val = os.path.join(os.path.join(data_path, filepath, val_filename+".pickle")) if val_filename else df_file_train

    bnn_smmc = BNN_smMC(model_name=filepath, list_param_names=params_list, train_set=df_file_train, val_set=df_file_val, 
        input_size=len(params_list), n_hidden=args.bnn_n_hidden)

    x_test, post_mean, post_std, evaluation_dict = bnn_smmc.run(n_epochs=args.bnn_n_epochs, lr=args.bnn_lr, 
        identifier=args.bnn_identifier, train_flag=False)
    pyro.clear_param_store()

    print("\n=== Loading GP model ===")

    out_filename = f"{args.gp_likelihood}_{train_filename}_epochs={args.gp_n_epochs}_lr={args.gp_lr}"

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

    ########
    # plot #
    ########
