import torch
import numpy as np 

def Poisson_satisfaction_function(lam):
    lam = lam.clone().detach()
    return torch.exp(-lam)*(1+lam+(lam**2)/2+(lam**3)/6)

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

