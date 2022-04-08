import sys
import time
import torch
import random
import numpy as np

sys.path.append(".")
from gpytorch.functions import log_normal_cdf
from GPs.utils import execution_time, normalize_columns

def posterior_predictive(model, x, n_trials, n_posterior_samples):
    Y_metadata = {'trials':1}

    normalized_x = normalize_columns(x).numpy()
    post_mean, post_std = model.predict(normalized_x, Y_metadata=Y_metadata)
    return post_mean.squeeze(), post_std.squeeze()

def train_GP(model, x_train, y_train):
    random.seed(0)
    np.random.seed(0)

    start = time.time()
    model.optimize()
    training_time = execution_time(start=start, end=time.time())

    print("\nTraining time =", training_time)
    return model, training_time

def evaluate_GP(model, n_posterior_samples, n_params, x_val=None, y_val=None, n_trials_val=None, z=1.96):
    random.seed(0)
    np.random.seed(0)

    n_val_points = len(x_val)
    
    start = time.time()
    post_mean, post_std = posterior_predictive(model=model, x=x_val, n_trials=n_trials_val, 
        n_posterior_samples=n_posterior_samples)
    evaluation_time = execution_time(start=start, end=time.time())

    val_satisfaction_prob = y_val.flatten().numpy()/n_trials_val
    assert val_satisfaction_prob.min()>=0
    assert val_satisfaction_prob.max()<=1

    val_dist = np.abs(val_satisfaction_prob-post_mean)
    n_val_errors = np.sum(val_dist > z*post_std)
    percentage_val_errors = 100*(n_val_errors/n_val_points)

    mse = np.mean(val_dist**2)
    mre = np.mean(val_dist/val_satisfaction_prob+0.000001)

    uncertainty_ci_area = 2*z*post_std
    avg_uncertainty_ci_area = np.mean(uncertainty_ci_area)

    print(f"Evaluation time = {evaluation_time}")
    print(f"Mean squared error: {mse}")
    # print(f"Mean relative error: {mre}")
    print(f"Percentage of validation errors: {percentage_val_errors} %")
    print(f"Average uncertainty area:  {avg_uncertainty_ci_area}\n")

    evaluation_dict = {"percentage_val_errors":percentage_val_errors, "mse":mse, "mre":mre, 
                    "avg_uncertainty_ci_area":avg_uncertainty_ci_area, "evaluation_time":evaluation_time}

    return x_val, post_mean, post_std, evaluation_dict  