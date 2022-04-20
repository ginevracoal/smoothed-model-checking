import os, sys
import argparse
import numpy as np
import pickle5 as pickle

sys.path.append(".")
from paths import *
from EP_GPs.smMC_GPEP import *
from plot_utils import plot_posterior

parser = argparse.ArgumentParser()
parser.add_argument("--load", default=False, type=eval, help="If True load the model else train it")
parser.add_argument("--n_epochs", default=1000, type=int, help="Max number of training iterations")
parser.add_argument("--lr", default=0.01, type=float, help="Learning rate")
parser.add_argument("--n_posterior_samples", default=10, type=int, help="Number of samples from posterior distribution")
args = parser.parse_args()

models_path = os.path.join("EP_GPs", models_path)
plots_path = os.path.join("EP_GPs", plots_path)

for filepath, train_filename, val_filename, params_list, math_params_list in case_studies:

    print(f"\n=== Training {train_filename} ===")

    out_filename = f"ep_gp_{train_filename}_epochs={args.n_epochs}_lr={args.lr}"

    with open(os.path.join(data_path, filepath, train_filename+".pickle"), 'rb') as handle:
        train_data = pickle.load(handle)
    
    paramterName = params_list[0] if len(params_list)==1 else ''.join(params_list)

    smc = smMC_GPEP()

    x_train, y_train, n_samples_train, n_trials_train = smc.transform_data(train_data)

    if args.load:
        smc.load(filepath=models_path, filename=out_filename)

    else:
        smc.fit(x_train, y_train, m_train)
        smc.save(filepath=models_path, filename=out_filename)

    print(f"\n=== Validation {val_filename} ===")

    try:
        with open(os.path.join(data_path, filepath, val_filename+".pickle"), 'rb') as handle:
            val_data = pickle.load(handle)
            
        x_val, y_val, n_samples_val, n_trials_val = smc.transform_data(val_data)

        post_mean, q1, q2, evaluation_dict = smc.eval_gp(x_train=x_train, x_val=x_val, y_val=val_data['labels'], 
            n_samples=n_samples_val, n_trials=n_trials_val)

        if len(params_list)<=2:

            fig = plot_posterior(params_list=params_list, math_params_list=math_params_list, train_data=train_data,
                test_data=val_data, val_data=val_data, post_mean=post_mean, q1=q1, q2=q2)

            os.makedirs(os.path.dirname(plots_path), exist_ok=True)
            fig.savefig(plots_path+f"{out_filename}.png")

    except:
        print("Validation set not available")