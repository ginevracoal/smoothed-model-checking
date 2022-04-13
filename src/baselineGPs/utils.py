import sys
import time
import torch
import random
import numpy as np

sys.path.append(".")
from gpytorch.functions import log_normal_cdf
from baselineGPs.binomial_likelihood import Binomial
from data_utils import normalize_columns, Poisson_observations, get_tensor_data
from evaluation_metrics import execution_time, evaluate_posterior_samples


def posterior_predictive(model, x, n_trials, n_posterior_samples):
    Y_metadata = {'trials':1}

    normalized_x = normalize_columns(x).numpy()

    model.Y_metadata=Y_metadata
    post_samples = model.posterior_samples(normalized_x, size=n_posterior_samples, likelihood=Binomial(), 
        Y_metadata=Y_metadata).squeeze()

    return post_samples

def train_GP(model, x_train, y_train):
    random.seed(0)
    np.random.seed(0)

    start = time.time()
    model.optimize("scg", max_iters=1000, messages=True)
    training_time = execution_time(start=start, end=time.time())

    print("\nTraining time =", training_time)
    return model, training_time

def evaluate_GP(model, n_posterior_samples, n_samples, val_data=None):
    random.seed(0)
    np.random.seed(0)


    if val_data is None: # Poisson case-study

        raise NotImplementedError

        # n_val_points = 100
        # x_val, y_val = Poisson_observations(n_val_points)
        # n_trials_val=1 

    else:
        x_val, y_val, n_samples, n_trials = get_tensor_data(val_data)

    start = time.time()
    post_samples = posterior_predictive(model=model, x=x_val, n_trials=n_trials, n_posterior_samples=n_posterior_samples)
    evaluation_time = execution_time(start=start, end=time.time())

    post_samples = np.transpose(post_samples)

    print(f"Evaluation time = {evaluation_time}")
    

    post_mean, q1, q2, evaluation_dict = evaluate_posterior_samples(y_val=y_val, post_samples=post_samples, n_samples=n_samples, 
        n_trials=n_trials)

    evaluation_dict.update({"evaluation_time":evaluation_time})
    return post_mean, q1, q2, evaluation_dict
