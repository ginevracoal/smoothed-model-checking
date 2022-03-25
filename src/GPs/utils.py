import torch
import numpy as np 


def normalize_columns(x, a=-1, b=1):
    """ Normalize columns of x in [a,b] range """
    min_x = torch.min(x, axis=0, keepdim=True)[0]
    max_x = torch.max(x, axis=0, keepdim=True)[0]
    normalized_x = 2*(x-min_x)/(max_x-min_x)-1
    return normalized_x

def Poisson_satisfaction_function(lam):
    lam = lam.clone().detach()
    return torch.exp(-lam)*(1+lam+(lam**2)/2+(lam**3)/6)

def Poisson_observations(n_points, n_params=1):
    x_val = []
    for col_idx in range(n_params):
        x_val.append(torch.linspace(0.1, 5, n_points))
    x_val = torch.stack(x_val, dim=1)
    y_val = Poisson_satisfaction_function(x_val)
    return x_val, y_val

def build_bernoulli_dataframe(data):
    params = torch.tensor(data['params'], dtype=torch.float32)
    params_observations = data['labels'].shape[1]

    params = np.repeat(params, params_observations, axis=0)
    labels = torch.tensor(data['labels'], dtype=torch.int64).flatten()
    
    print("\nparams shape =", params.shape)
    print("labels shape =", labels.shape)

    n_params = data['params'].shape[1]
    return params, labels, n_params

def build_binomial_dataframe(data):
    params = torch.tensor(data['params'], dtype=torch.float32)
    labels = torch.tensor(data['labels'], dtype=torch.int)

    n_params = params.shape[1]
    n_trials = labels.shape[1]

    success_counts = [len(row[row==1.]) for row in labels]
    success_counts = torch.tensor(success_counts, dtype=torch.float32)

    print("\nparams shape =", params.shape)
    print("labels shape =", labels.shape)
    print("n. trials =", n_trials)
    print("Params True label counts shape =", success_counts.shape)

    return params, success_counts, n_params, n_trials

def execution_time(start, end):
    hours, rem = divmod(end - start, 3600)
    minutes, seconds = divmod(rem, 60)
    print("\nExecution time = {:0>2}:{:0>2}:{:0>2}".format(int(hours), int(minutes), int(seconds)))

