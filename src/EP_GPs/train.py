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

    with open(os.path.join(data_path, filepath, train_filename+".pickle"), 'rb') as handle:
        train_data = pickle.load(handle)
    
    paramterName = params_list[0] if len(params_list)==1 else ''.join(params_list)

    x_train = train_data["params"]
    p_train = train_data["labels"]
    n_train_points, m_train = p_train.shape
    y_train = np.sum(p_train,axis=1)/m_train
    y_train = np.expand_dims(y_train,axis=1)

    with open(os.path.join(data_path, filepath, val_filename+".pickle"), 'rb') as handle:
        val_data = pickle.load(handle)
        
    x_val = val_data["params"]
    p_test = val_data["labels"]       
    n_test_points, m_test = p_test.shape
    y_test = np.sum(p_test,axis=1)/m_test
    y_test = np.expand_dims(y_test,axis=1)

    smc = smMC_GPEP()

    if args.load:

        smc.load(filepath=models_path, filename=train_filename)
        post_mean, q1, q2 = smc.make_predictions(x_train=x_train, x_test=x_val)
        # post_mean, q1, q2, evaluation_dict = smc.eval_gp(val_data)

    else:

        smc.fit(x_train, y_train, m_train)
        smc.save(filepath=models_path, filename=train_filename)

        post_mean, q1, q2 = smc.make_predictions(x_train=x_train, x_test=x_val)
        # post_mean, q1, q2, evaluation_dict = smc.eval_gp(val_data)


    if len(params_list)<=2:

        fig = plot_posterior(params_list=params_list, math_params_list=math_params_list, train_data=train_data,
            test_data=val_data, val_data=val_data, post_mean=post_mean, q1=q1, q2=q2)

        os.makedirs(os.path.dirname(plots_path), exist_ok=True)
        fig.savefig(plots_path+f"{val_filename}.png")
