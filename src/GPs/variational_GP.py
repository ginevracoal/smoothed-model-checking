import sys
import time
import torch
import random
import gpytorch
import numpy as np
import pandas as pd
import seaborn as sns
from itertools import product
import matplotlib.pyplot as plt
from gpytorch.models import ApproximateGP
from gpytorch.functions import log_normal_cdf
from gpytorch.variational import MeanFieldVariationalDistribution, CholeskyVariationalDistribution
from gpytorch.variational import VariationalStrategy, UnwhitenedVariationalStrategy

from utils import execution_time, Poisson_satisfaction_function, normalize_columns


class GPmodel(ApproximateGP):

    def __init__(self, inducing_points, variational_distribution='cholesky', variational_strategy='default'):

        if variational_distribution=='cholesky':
            variational_distribution = CholeskyVariationalDistribution(inducing_points.size(0))
        elif variational_distribution=='meanfield':
            variational_distribution = MeanFieldVariationalDistribution(inducing_points.size(0))
        else:
            raise NotImplementedError

        if variational_strategy=='default':
            variational_strategy = VariationalStrategy(self, inducing_points, variational_distribution, 
                                                                learn_inducing_locations=False)
        elif variational_strategy=='unwhitened':
            variational_strategy = UnwhitenedVariationalStrategy(self, inducing_points, variational_distribution, 
                                                                learn_inducing_locations=False)
        else:
            raise NotImplementedError

        super(GPmodel, self).__init__(variational_strategy)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())

    def forward(self, x):
        mean = self.mean_module(x)
        covar = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean, covar)



def train_GP(model, likelihood, x_train, y_train, n_trials_train, n_epochs, lr):
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)

    model.train()
    likelihood.train()
    likelihood.n_trials = n_trials_train

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    elbo = gpytorch.mlls.VariationalELBO(likelihood, model, num_data=len(x_train))

    print()
    start = time.time()
    for i in range(n_epochs):
        optimizer.zero_grad()
        output = model(x_train)
        loss = -elbo(output, y_train)
        loss.backward()
        optimizer.step()

        if i % 10 == 0:
            print(f"Epoch {i}/{n_epochs} - Loss: {loss}")

    execution_time(start=start, end=time.time())

    print("\nModel params:", model.state_dict().keys())
    return model

def evaluate_GP(model, likelihood, n_posterior_samples, n_params, x_val=None, y_val=None, n_trials_val=None,
    n_test_points=None, z=1.96):
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
        
    model.eval()    
    likelihood.eval()

    with torch.no_grad():

        if x_val is None:

            x_test = []
            for col_idx in range(n_params):
                x_test.append(torch.linspace(0.1, 5, n_test_points))
            x_test = torch.stack(x_test, dim=1)

            x_val = x_test
            y_val = Poisson_satisfaction_function(x_test)
            n_trials_val = 1
            n_test_points_samples = n_test_points

        else:

            if n_test_points is None:

                n_test_points = len(x_val)
                x_test = x_val
                n_test_points_samples = n_test_points

            else:
                x_test = []
                for col_idx in range(n_params):
                    single_param_values = x_val[:,col_idx]
                    x_test.append(torch.linspace(single_param_values.min(), single_param_values.max(), n_test_points))
                x_test = torch.stack(x_test, dim=1)
                n_test_points_samples = n_test_points**n_params

                if n_params==2:
                    x_test = torch.tensor(list(product(x_test[:,0], x_test[:,1])))
        
        normalized_x_test = normalize_columns(x_test) 
        posterior = model(normalized_x_test)
        post_samples = posterior.sample(sample_shape=torch.Size((n_posterior_samples,)))
        post_samples = [[torch.exp(log_normal_cdf(post_samples[j, i])) for i in range(n_test_points_samples)] \
            for j in range(n_posterior_samples)]
        post_samples = torch.tensor(post_samples)

        post_mean = torch.mean(post_samples, dim=0).flatten()
        post_std = torch.std(post_samples, dim=0).flatten()

        if len(x_val)==len(x_test):

            val_satisfaction_prob = y_val.flatten()/n_trials_val
            val_dist = torch.abs(val_satisfaction_prob-post_mean)
            # print("\nval_satisfaction_prob.shape", val_satisfaction_prob.shape)
            # print("val_dist", val_dist)
            # print("val_satisfaction_prob", val_satisfaction_prob)

            n_val_errors = torch.sum(val_dist > z*post_std)
            percentage_val_errors = 100*(n_val_errors/n_test_points)

            mse = torch.mean(val_dist**2)
            mre = torch.mean(val_dist/val_satisfaction_prob+0.000001)

            UncVolume = 2*z*post_std
            AvgUncVolume = torch.mean(UncVolume)

            print(f"\nPercentage of validation errors = {percentage_val_errors}")
            print(f"MSE = {mse}")
            print(f"MRE = {mre}")
            print(f"AvgUncVolume = {AvgUncVolume}")

    return x_test, post_mean, post_std


def plot_GP_posterior(x_train_binomial, y_train_binomial, n_trials_train, x_test, post_mean, post_std, 
    params_list, x_val=None, y_val=None, n_trials_val=None, z=1.96):

    n_params = len(params_list)

    if n_params==1:

        fig, ax = plt.subplots(1, 1, figsize=(8, 5))
        
        if x_val is None:

            sns.lineplot(x=x_test.flatten(), y=post_mean, ax=ax, label='posterior')
            ax.fill_between(x_test.flatten(), post_mean-z*post_std, post_mean+z*post_std, alpha=0.5)

            sns.lineplot(x=x_test.flatten(), y=Poisson_satisfaction_function(x_test).flatten(), ax=ax, 
                label='true satisfaction')
            sns.scatterplot(x=x_train_binomial.flatten(), y=y_train_binomial.flatten()/n_trials_train, ax=ax, 
                label='training points', marker='.', color='black')

        else:
            sns.scatterplot(x=x_val.flatten(), y=y_val.flatten()/n_trials_val, ax=ax, label='validation pts')
            sns.scatterplot(x=x_train_binomial.flatten(), y=y_train_binomial.flatten()/n_trials_train, ax=ax, 
                label='training points', marker='.', color='black')
            sns.lineplot(x=x_test.flatten(), y=post_mean, ax=ax, label='posterior')
            ax.fill_between(x_test.flatten(), post_mean-z*post_std, post_mean+z*post_std, alpha=0.5)

    elif n_params==2:

        p1, p2 = params_list

        fig, ax = plt.subplots(1, 2, figsize=(12, 5))

        data = pd.DataFrame({p1:x_val[:,0],p2:x_val[:,1],'val_counts':y_val.flatten()/n_trials_val})
        data[p1] = data[p1].apply(lambda x: format(float(x),".4f"))
        data[p2] = data[p2].apply(lambda x: format(float(x),".4f"))
        data = data.pivot(p1, p2, "val_counts")
        sns.heatmap(data, ax=ax[0], label='validation pts')
        ax[0].set_xlabel("validation set")

        # data = pd.DataFrame({p1:x_test[:,0],p2:x_test[:,1],'posterior_mean':posterior.mean})
        data = pd.DataFrame({p1:x_test[:,0],p2:x_test[:,1],'posterior_preds':post_mean})

        data[p1] = data[p1].apply(lambda x: format(float(x),".4f"))
        data[p2] = data[p2].apply(lambda x: format(float(x),".4f"))
        data = data.pivot(p1, p2, "posterior_preds")
        data.sort_index(level=0, ascending=True, inplace=True)

        sns.heatmap(data, ax=ax[1], label='posterior preds')
        ax[1].set_xlabel("posterior preds")

    plt.tight_layout()
    plt.close()
    return fig