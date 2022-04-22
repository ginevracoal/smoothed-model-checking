import time
import torch
import numpy
import numpy as np

def intervals_intersection(a,b):
    min_right = np.minimum(a[1],b[1])
    max_left = np.maximum(a[0],b[0])
    return np.maximum(0, min_right-max_left)

def evaluate_posterior_samples(y_val, post_samples, n_samples, n_trials, z=1.96, alpha1=0.025, alpha2=0.975):

    if y_val.shape != (n_samples, n_trials):
        raise ValueError("y_val should be bernoulli trials")

    if type(post_samples)==torch.Tensor:
        post_samples = post_samples.detach().numpy()

    if type(y_val)==torch.Tensor:
        y_val = y_val.detach().numpy()

    satisfaction_prob = y_val.mean(1).flatten()
    assert satisfaction_prob.min()>=0
    assert satisfaction_prob.max()<=1

    post_mean = post_samples.mean(0).squeeze()
    # post_std = post_samples.std(0).squeeze()
    print(satisfaction_prob.shape, post_mean.shape)
    assert satisfaction_prob.shape == post_mean.shape

    q1, q2 = np.quantile(post_samples, q=[alpha1, alpha2], axis=0)
    assert satisfaction_prob.shape == q1.shape

    sample_variance = [((param_y-param_y.mean())**2).mean() for param_y in y_val]
    val_std = np.sqrt(sample_variance).flatten()
    validation_ci = (satisfaction_prob-(z*val_std)/np.sqrt(n_trials),satisfaction_prob+(z*val_std)/np.sqrt(n_trials))
    estimated_ci = (q1,q2)
    non_empty_intersections = np.sum(intervals_intersection(validation_ci,estimated_ci)>0)
    val_accuracy = 100*non_empty_intersections/n_samples
    assert val_accuracy <= 100

    val_dist = np.abs(satisfaction_prob-post_mean)
    mse = np.mean(val_dist**2)
    # mre = np.mean(val_dist/satisfaction_prob+0.000001)

    ci_uncertainty_area = q2-q1
    avg_uncertainty_area = np.mean(ci_uncertainty_area)

    print(f"Mean squared error: {mse}")
    # print(f"Mean relative error: {mre}")
    print(f"Validation accuracy: {val_accuracy} %")
    print(f"Average uncertainty area:  {avg_uncertainty_area}\n")

    evaluation_dict = {"val_accuracy":val_accuracy, "mse":mse, "avg_uncertainty_area":avg_uncertainty_area}

    return post_mean, q1, q2, evaluation_dict

def execution_time(start, end):
    hours, rem = divmod(end - start, 3600)
    minutes, seconds = divmod(rem, 60)
    time = f"{int(hours):0>2}:{int(minutes):0>2}:{int(seconds):05.4f}"
    return time
