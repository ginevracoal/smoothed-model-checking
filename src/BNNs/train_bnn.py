import os
import sys
import pyro
import torch
import random
import argparse
import numpy as np

sys.path.append(".")
from BNNs.bnn import BNN_smMC
from paths import data_paths, data_path

random.seed(0)
np.random.seed(0)
torch.manual_seed(0)


parser = argparse.ArgumentParser()
parser.add_argument("--train", default=True, type=eval)
parser.add_argument("--architecture", default='2L', type=str)
parser.add_argument("--n_epochs", default=10000, type=int, help="Number of training iterations")
parser.add_argument("--lr", default=0.01, type=float, help="Learning rate")
parser.add_argument("--identifier", default=1, type=int)
parser.add_argument("--n_hidden", default=10, type=int)
args = parser.parse_args()



for filepath, train_filename, val_filename, params_list in data_paths:

    df_file_train = os.path.join(os.path.join(data_path, filepath, train_filename+".pickle"))
    df_file_val = os.path.join(os.path.join(data_path, filepath, val_filename+".pickle")) if val_filename else df_file_train

    print("TrainFlag = ", args.train)
    print("Train set: ", df_file_train)
    print("Validation set: ", df_file_val)
    print(f"\nmodel_name = {filepath}, param_name = {''.join(params_list)},\
        n_hidden = {args.n_hidden}, n_epochs = {args.n_epochs}, lr = {args.lr}, identifier = {args.identifier}")

    bnn_smmc = BNN_smMC(model_name=filepath, list_param_names=params_list, train_set=df_file_train, val_set=df_file_val, 
        input_size=len(params_list), n_hidden=args.n_hidden, architecture_name=args.architecture)

    bnn_smmc.run(n_epochs=args.n_epochs, lr=args.lr, identifier=args.identifier, train_flag=args.train)
    pyro.clear_param_store()
