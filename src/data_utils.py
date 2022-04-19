import torch
import numpy as np 


def normalize_columns(x, a=-1, b=1, min_x=None, max_x=None, return_minmax=False):
    """ Normalize columns of x in [a,b] range """
    if min_x is None:
        min_x = torch.min(x, axis=0, keepdim=True)[0]

    if max_x is None:
        max_x = torch.max(x, axis=0, keepdim=True)[0]

    normalized_x = 2*(x-min_x)/(max_x-min_x)-1

    if return_minmax:
        return min_x, max_x, normalized_x
    else:
        return normalized_x

def Poisson_satisfaction_function(lam):
    if type(lam)==np.ndarray:
        return np.exp(-lam)*(1+lam+(lam**2)/2+(lam**3)/6)

    elif type(lam)==torch.Tensor:
        lam = lam.clone().detach()
        return torch.exp(-lam)*(1+lam+(lam**2)/2+(lam**3)/6)

    else:
        raise NotImplementedError

def Poisson_observations(n_points, n_params=1):
    x_val = []
    for col_idx in range(n_params):
        x_val.append(torch.linspace(0.1, 5, n_points))
    x_val = torch.stack(x_val, dim=1)
    y_val = Poisson_satisfaction_function(x_val).squeeze()
    return x_val, y_val

def get_tensor_data(data, verbose=False):

    x_data = torch.tensor(data['params'], dtype=torch.float32)
    y_data = torch.tensor(data['labels'], dtype=torch.float32)
    n_samples = len(x_data)
    n_trials = y_data.shape[1]

    if verbose:
        print("\nx_data shape =", x_data.shape)
        print("y_data shape =", y_data.shape)
        print("n_samples =", n_samples)
        print("n_trials =", n_trials)

    return x_data, y_data, n_samples, n_trials

def get_bernoulli_data(data, verbose=False):

    x_data, y_data, n_samples, n_trials = get_tensor_data(data, verbose=verbose)
    params = np.repeat(x_data, n_trials, axis=0)
    y_data = y_data.flatten()

    return x_data, y_data, n_samples, n_trials

def get_binomial_data(data, verbose=False):

    x_data, y_data, n_samples, n_trials = get_tensor_data(data, verbose=verbose)
    success_counts = [len(row[row==1.]) for row in y_data]
    y_data = torch.tensor(success_counts, dtype=torch.float32)

    return x_data, y_data, n_samples, n_trials


