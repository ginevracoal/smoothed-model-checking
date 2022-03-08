import os
import sys
import torch
import gpytorch
import numpy as np
import pandas as pd
import pickle as pkl
import seaborn as sns
import matplotlib.pyplot as plt
import torch.utils.data as data_utils
from data import split_train_test, squash_df
from binomial_likelihood import BinomialLikelihood

from gpytorch.models import ApproximateGP
from gpytorch.variational import VariationalStrategy
from gpytorch.variational import MeanFieldVariationalDistribution

import pickle5 as pickle


class GPmodel(ApproximateGP):

    def __init__(self, inducing_points):
        variational_distribution = MeanFieldVariationalDistribution(inducing_points.size(0))
        variational_strategy = VariationalStrategy(self, inducing_points, variational_distribution, 
                                                    learn_inducing_locations=True)
        super(GPmodel, self).__init__(variational_strategy)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())

    def forward(self, x):
        mean = self.mean_module(x)
        covar = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean, covar)

for dirname, filename in [
    # ["Poisson","Poisson_DS_46samples_1obs_Lambda"],
    # ["Poisson","Poisson_DS_46samples_5obs_Lambda"],
    # ["Poisson","Poisson_DS_46samples_10obs_Lambda"],
    ["SIR", "SIR_DS_20samples_5000obs_Beta"],
    ["SIR", "SIR_DS_20samples_5000obs_Gamma"],
    ["SIR", "SIR_DS_200samples_10obs_Beta"],
    ["SIR", "SIR_DS_200samples_10obs_Gamma"],
    ["SIR", "SIR_DS_256samples_10obs_BetaGamma"],
    ["SIR", "SIR_DS_256samples_5000obs_BetaGamma"],
    ]:

    print(f"\n=== {filename} ===")

    with open(f"../Data/{dirname}/{filename}.pickle", 'rb') as handle:
        data = pickle.load(handle)

    params = torch.tensor(data['params'], dtype=torch.float32)
    labels = torch.tensor(data['labels'], dtype=torch.int)

    print("\nparams shape =", params.shape)
    print("labels shape =", labels.shape)
    print("n. unique params =", len(np.unique(params)))
    print("\nRow-wise True label counts:\n", [len(row[row==1.]) for row in labels])

    success_counts = [len(row[row==1.]) for row in labels]

    n_trials = labels.shape[1]

    params_tensor = params
    labels_tensor = torch.tensor(success_counts, dtype=torch.float32)
    train = data_utils.TensorDataset(params_tensor, labels_tensor)
    train_loader = data_utils.DataLoader(train, batch_size=10, shuffle=True)

    model = GPmodel(inducing_points=params_tensor)
    likelihood = BinomialLikelihood(n_trials=n_trials)

    model.train()
    likelihood.train()

    optimizer = torch.optim.Adam([{'params': model.parameters()}, {'params': likelihood.parameters()}], lr=0.01)
    elbo = gpytorch.mlls.VariationalELBO(likelihood, model, num_data=len(labels_tensor))

    num_epochs = 30

    for i in range(num_epochs):
        for x_batch, y_batch in train_loader:
            optimizer.zero_grad()
            output = model(x_batch)
            loss = -elbo(output, y_batch)
            loss.backward()
            optimizer.step()
        print(f"Epoch {i}/{num_epochs} - Loss: {loss}")

    print("\nModel params:", model.state_dict().keys())

    os.makedirs(os.path.dirname('models/'), exist_ok=True)
    torch.save(model.state_dict(), f'models/gp_state_{filename}.pth')

    state_dict = torch.load(f'models/gp_state_{filename}.pth')
    model.load_state_dict(state_dict)

    model.eval()    
    likelihood.eval()

    with torch.no_grad():
        x_test = []
        for col_idx in range(params_tensor.shape[1]):
            single_param_values = params_tensor[:,col_idx]
            x_test.append(torch.linspace(single_param_values.min(), single_param_values.max(), 100))

        x_test = torch.stack(x_test, dim=1)
        observed_pred = likelihood(model(x_test))
        pred_labels = observed_pred.mean
        pred_variance = observed_pred.variance

    path='plots/'+dirname+'/'
    os.makedirs(os.path.dirname(path), exist_ok=True)

    n_params = params_tensor.shape[1]
    
    fig, ax = plt.subplots(1, n_params, figsize=(6*n_params, 5))

    for col_idx in range(params_tensor.shape[1]):
        single_param_values = params_tensor[:,col_idx]
        single_param_x_test = x_test[:,col_idx]

        axis = ax if n_params==1 else ax[col_idx]
        sns.scatterplot(x=single_param_values, y=labels_tensor, ax=axis)
        sns.lineplot(x=single_param_x_test, y=pred_labels, ax=axis)
        axis.fill_between(single_param_x_test.numpy(), pred_labels-pred_variance, pred_labels+pred_variance, 
                        alpha=0.5)
    
    fig.savefig(path+f"lineplot_{filename}.png")
    plt.close()


