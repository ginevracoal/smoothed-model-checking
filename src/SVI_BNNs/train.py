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
from posterior_plot_utils import plot_posterior

random.seed(0)
np.random.seed(0)
torch.manual_seed(0)


parser = argparse.ArgumentParser()
parser.add_argument("--likelihood", default='binomial', type=str, help="Choose 'bernoulli' or 'binomial'")
parser.add_argument("--architecture", default='3L', type=str, help="NN architecture")
parser.add_argument("--batch_size", default=100, type=int, help="")
parser.add_argument("--n_epochs", default=5000, type=int, help="Number of training iterations")
parser.add_argument("--lr", default=0.001, type=float, help="Learning rate")
parser.add_argument("--n_hidden", default=30, type=int, help="Size of hidden layers")
parser.add_argument("--n_posterior_samples", default=100, type=int, help="Number of samples from posterior distribution")
parser.add_argument("--device", default="cpu", type=str, help="Choose 'cpu' or 'cuda'")
parser.add_argument("--load", default=False, type=eval)
args = parser.parse_args()

plots_path = os.path.join(plots_path, "SVI_BNNs/")
models_path = os.path.join(models_path, "SVI_BNNs/")


for filepath, train_filename, val_filename, params_list, math_params_list in case_studies:

    if len(params_list)==6:
        args.n_epochs = 100
        args.batch_size = 5000

    print(args)

    print(f"\n=== SVI BNN Training {train_filename} ===")

    out_filename = f"svi_bnn_{train_filename}_epochs={args.n_epochs}_lr={args.lr}_batch={args.batch_size}_hidden={args.n_hidden}_{args.architecture}"

    with open(os.path.join(data_path, filepath, train_filename+".pickle"), 'rb') as handle:
        train_data = pickle.load(handle)

    bnn_smmc = BNN_smMC(model_name=filepath, list_param_names=params_list, likelihood=args.likelihood,
        input_size=len(params_list), n_hidden=args.n_hidden, architecture_name=args.architecture)

    if args.load:
        bnn_smmc.load(filepath=models_path, filename=out_filename, device=args.device)
    else:
        bnn_smmc.train(train_data=train_data, n_epochs=args.n_epochs, lr=args.lr, batch_size=args.batch_size,
            device=args.device)
        bnn_smmc.save(filepath=models_path, filename=out_filename, training_device=args.device)

    print(f"\n=== SVI BNN Validation {val_filename} ===")

    with open(os.path.join(data_path, filepath, val_filename+".pickle"), 'rb') as handle:
        val_data = pickle.load(handle)

    post_mean, q1, q2, evaluation_dict = bnn_smmc.evaluate(train_data=train_data, val_data=val_data,
        n_posterior_samples=args.n_posterior_samples, device=args.device)

    if len(params_list)<=2:

        fig = plot_posterior(params_list=params_list, math_params_list=math_params_list, train_data=train_data,
            test_data=val_data, val_data=val_data, post_mean=post_mean, q1=q1, q2=q2)

        os.makedirs(os.path.dirname(plots_path), exist_ok=True)
        fig.savefig(plots_path+f"{out_filename}.png")

    pyro.clear_param_store()