import os
import sys
import GPy
import torch
import random
import argparse
import numpy as np
from math import sqrt
import pickle5 as pickle

sys.path.append(".")
from paths import *
from baselineGPs.utils import train_GP, evaluate_GP
from baselineGPs.binomial_likelihood import Binomial
from GPs.utils import build_bernoulli_dataframe, build_binomial_dataframe, normalize_columns

random.seed(0)
np.random.seed(0)


parser = argparse.ArgumentParser()
parser.add_argument("--likelihood", default='binomial', type=str, help='')
parser.add_argument("--load", default=False, type=eval, help="If True load the model else train it")
parser.add_argument("--n_posterior_samples", default=50, type=int, help="Number of samples from posterior distribution")
args = parser.parse_args()


models_path = os.path.join("baselineGPs", models_path)
os.makedirs(os.path.dirname(models_path), exist_ok=True)

for filepath, train_filename, val_filename, params_list, math_params_list in data_paths:

    print(f"\n=== Training {train_filename} ===")

    with open(os.path.join(data_path, filepath, train_filename+".pickle"), 'rb') as handle:
        data = pickle.load(handle)

    if args.likelihood=='binomial':
        x_train, y_train, n_params, n_trials_train = build_binomial_dataframe(data)
        x_train_binomial, y_train_binomial = x_train, y_train

    else:
        raise NotImplementedError

    normalized_x_train = normalize_columns(x_train).numpy()
    y_train = y_train.unsqueeze(1).numpy()

    out_filename = f"{train_filename}"
    Y_metadata = {'trials':np.full(y_train.shape, n_trials_train)}
    
    likelihood = Binomial()
    kernel = GPy.kern.RBF(input_dim=n_params, variance=1., lengthscale=0.5)
    inference = GPy.inference.latent_function_inference.Laplace()
    # model = GPy.models.GPClassification(X=normalized_x_train, Y=y_train, kernel=kernel, 
    #                                 likelihood=likelihood, Y_metadata=Y_metadata)
    model = GPy.core.GP(X=normalized_x_train, Y=y_train, kernel=kernel, inference_method=inference,
                                    likelihood=likelihood, Y_metadata=Y_metadata)

    if args.load:
        with open(os.path.join(models_path, "gp_"+out_filename+".pkl"), 'rb') as file:
            model = pickle.load(file)

        file = open(os.path.join(models_path,f"gp_{out_filename}_training_time.txt"),"r+")
        print(f"\nTraining time = {file.read()}")

    else:

        model, training_time = train_GP(model=model, x_train=normalized_x_train, y_train=y_train)

        with open(os.path.join(models_path, "gp_"+out_filename+".pkl"), 'wb') as file:
            pickle.dump(model, file)

        file = open(os.path.join(models_path,f"gp_{out_filename}_training_time.txt"),"w")
        file.writelines(training_time)
        file.close()

    print(f"\n=== Validation {val_filename} ===")

    if filepath=='Poisson':
        raise NotImplementedError

    else:
        with open(os.path.join(data_path, filepath, val_filename+".pickle"), 'rb') as handle:
            val_data = pickle.load(handle)
        
        x_val, y_val, n_params, n_trials_val = build_binomial_dataframe(val_data)

        x_test, post_mean, post_std, evaluation_dict = evaluate_GP(model=model, x_val=x_val, y_val=y_val, 
            n_trials_val=n_trials_val, n_posterior_samples=args.n_posterior_samples, n_params=n_params)
