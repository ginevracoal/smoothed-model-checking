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
from SVI_BNNs.bnn import BNN_smMC
from plot_utils import plot_posterior


###
from SVI_BNNs.dnn import DeterministicNetwork
from data_utils import *
from torch.utils.data import TensorDataset, DataLoader

#####

random.seed(0)
np.random.seed(0)
torch.manual_seed(0)


parser = argparse.ArgumentParser()
parser.add_argument("--likelihood", default='binomial', type=str, help="Choose 'bernoulli' or 'binomial'")
parser.add_argument("--architecture", default='3L', type=str, help="NN architecture")
parser.add_argument("--batch_size", default=500, type=int, help="")
parser.add_argument("--n_epochs", default=10000, type=int, help="Number of training iterations")
parser.add_argument("--lr", default=0.01, type=float, help="Learning rate")
parser.add_argument("--n_hidden", default=10, type=int, help="Size of hidden layers")
parser.add_argument("--n_posterior_samples", default=100, type=int, help="Number of samples from posterior distribution")
parser.add_argument("--device", default="cpu", type=str, help="Choose 'cpu' or 'cuda'")
parser.add_argument("--load", default=False, type=eval)
args = parser.parse_args()
print(args)

plots_path = os.path.join(plots_path, "SVI_BNNs/")
models_path = os.path.join(models_path, "SVI_BNNs/")


for filepath, train_filename, val_filename, params_list, math_params_list in case_studies:

    print(f"\n=== SVI BNN Training {train_filename} ===")

    out_filename = f"svi_bnn_{train_filename}_epochs={args.n_epochs}_lr={args.lr}_batch={args.batch_size}_hidden={args.n_hidden}_{args.architecture}"

    with open(os.path.join(data_path, filepath, train_filename+".pickle"), 'rb') as handle:
        train_data = pickle.load(handle)


    ###############
    # dnn = DeterministicNetwork(input_size=len(params_list), hidden_size=args.n_hidden, 
    #         architecture_name=args.architecture)

    # if args.likelihood=='bernoulli':
    #     x_train, y_train, n_samples, n_trials_train = get_bernoulli_data(train_data)
        
    # elif args.likelihood=='binomial':
    #     x_train, y_train, n_samples, n_trials_train = get_binomial_data(train_data)

    # else:
    #     raise AttributeError

    # x_train = normalize_columns(x_train)
    # y_train = y_train.unsqueeze(1) # check this

    # dataset = TensorDataset(x_train, y_train) 
    # train_loader = DataLoader(dataset=dataset, batch_size=args.batch_size, shuffle=True)

    # dnn.train(train_loader=train_loader, n_trials_train=n_trials_train, epochs=args.n_epochs, lr=args.lr, likelihood=args.likelihood, device=args.device)

    # with open(os.path.join(data_path, filepath, val_filename+".pickle"), 'rb') as handle:
    #     val_data = pickle.load(handle)

    # predictions = dnn.evaluate(train_data=train_data, val_data=val_data, device=args.device)

    # if len(params_list)<=2:

    #     fig = plot_posterior(params_list=params_list, math_params_list=math_params_list, train_data=train_data,
    #         test_data=val_data, val_data=val_data, post_mean=predictions, q1=predictions, q2=predictions)

    #     os.makedirs(os.path.dirname(plots_path), exist_ok=True)
    #     fig.savefig(plots_path+f"{out_filename}.png")

    # exit()
    ##############

    bnn_smmc = BNN_smMC(model_name=filepath, list_param_names=params_list, likelihood=args.likelihood,
        input_size=len(params_list), n_hidden=args.n_hidden, architecture_name=args.architecture)

    if args.load:
        bnn_smmc.load(filepath=models_path, filename=out_filename)
    else:
        bnn_smmc.train(train_data=train_data, n_epochs=args.n_epochs, lr=args.lr, batch_size=args.batch_size,
            device=args.device)
        bnn_smmc.save(filepath=models_path, filename=out_filename)

    print(f"\n=== SVI BNN Validation {val_filename} ===")

    try:

        with open(os.path.join(data_path, filepath, val_filename+".pickle"), 'rb') as handle:
            val_data = pickle.load(handle)

        post_mean, q1, q2, evaluation_dict = bnn_smmc.evaluate(train_data=train_data, val_data=val_data,
            n_posterior_samples=args.n_posterior_samples, device=args.device)

        if len(params_list)<=2:

            fig = plot_posterior(params_list=params_list, math_params_list=math_params_list, train_data=train_data,
                test_data=val_data, val_data=val_data, post_mean=post_mean, q1=q1, q2=q2)

            os.makedirs(os.path.dirname(plots_path), exist_ok=True)
            fig.savefig(plots_path+f"{out_filename}.png")

    except:
        print("Validation set not available")

    pyro.clear_param_store()