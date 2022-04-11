import time
import torch
import numpy
import numpy as np


def evaluate_posterior_samples(y, post_samples, n_params, n_trials): # z=1.96

    if type(post_samples)==numpy.ndarray:

        satisfaction_prob = y/n_trials
        assert satisfaction_prob.min()>=0
        assert satisfaction_prob.max()<=1

        post_mean = post_samples.mean(0).squeeze()
        post_std = post_samples.std(0).squeeze()
        assert satisfaction_prob.shape == post_mean.shape

        q1, q2 = np.quantile(post_samples, q=[0.05, 0.95], axis=0)
        assert satisfaction_prob.shape == q1.shape

        n_val_errors = np.sum(satisfaction_prob < q1) + np.sum(satisfaction_prob > q2)
        percentage_val_errors = 100*(n_val_errors/n_params)
        assert percentage_val_errors < 100

        val_dist = np.abs(satisfaction_prob-post_mean)
        mse = np.mean(val_dist**2)
        mre = np.mean(val_dist/satisfaction_prob+0.000001)

        ci_uncertainty_area = q2-q1 #2*z*post_std
        avg_uncertainty_area = np.mean(ci_uncertainty_area)

    elif type(post_samples)==torch.Tensor:

        satisfaction_prob = y/n_trials
        assert satisfaction_prob.min()>=0
        assert satisfaction_prob.max()<=1

        post_mean = torch.mean(post_samples, dim=0).flatten()
        post_std = torch.std(post_samples, dim=0).flatten()
        assert satisfaction_prob.shape == post_mean.shape

        q1, q2 = torch.quantile(post_samples, q=torch.tensor([0.05, 0.95]), dim=0)
        assert satisfaction_prob.shape == q1.shape 

        n_val_errors = torch.sum(satisfaction_prob<q1) + torch.sum(satisfaction_prob>q2)
        percentage_val_errors = 100*n_val_errors/n_params
        assert percentage_val_errors < 100

        val_dist = torch.abs(satisfaction_prob-post_mean)
        mse = torch.mean(val_dist**2)
        mre = torch.mean(val_dist/satisfaction_prob+0.000001)

        ci_uncertainty_area = q2-q1 #2*z*post_std
        avg_uncertainty_area = torch.mean(ci_uncertainty_area)

    else: 
        raise NotImplementedError

    print(f"Mean squared error: {mse}")
    # print(f"Mean relative error: {mre}")
    print(f"Percentage of validation errors: {percentage_val_errors} %")
    print(f"Average uncertainty area:  {avg_uncertainty_area}\n")

    evaluation_dict = {"percentage_val_errors":percentage_val_errors, "mse":mse, "mre":mre, 
                       "avg_uncertainty_area":avg_uncertainty_area}

    return post_mean, post_std, q1, q2, evaluation_dict

def execution_time(start, end):
    hours, rem = divmod(end - start, 3600)
    minutes, seconds = divmod(rem, 60)
    time = f"{int(hours):0>2}:{int(minutes):0>2}:{int(seconds):0>2}"
    return time
