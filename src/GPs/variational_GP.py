import sys
import time
import torch
import random
import gpytorch
import itertools
import numpy as np
import pandas as pd
import seaborn as sns
from tqdm import tqdm
from itertools import product
import matplotlib.pyplot as plt
from gpytorch.models import ApproximateGP
from gpytorch.functions import log_normal_cdf
from torch.utils.data import TensorDataset, DataLoader
from gpytorch.variational import MeanFieldVariationalDistribution, CholeskyVariationalDistribution
from gpytorch.variational import VariationalStrategy, UnwhitenedVariationalStrategy

sys.path.append(".")
from GPs.utils import execution_time, normalize_columns, Poisson_satisfaction_function, Poisson_observations


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

    dataset = TensorDataset(x_train,y_train) 
    train_loader = DataLoader(dataset=dataset, batch_size=1000, shuffle=True)

    model.train()
    likelihood.train()
    likelihood.n_trials = n_trials_train

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    elbo = gpytorch.mlls.VariationalELBO(likelihood, model, num_data=len(x_train))

    print()
    start = time.time()
    for i in tqdm(range(n_epochs)):
        for x_batch, y_batch in train_loader:
            optimizer.zero_grad()
            output = model(x_batch)
            loss = -elbo(output, y_batch)
            loss.backward()
            optimizer.step()

        if i % 10 == 0:
            print(f"Epoch {i}/{n_epochs} - Loss: {loss}")

    training_time = execution_time(start=start, end=time.time())
    print("\nTraining time =", training_time)

    print("\nModel params:", model.state_dict().keys())
    return model, training_time

def posterior_predictive(model, x, n_posterior_samples):
    normalized_x = normalize_columns(x) 
    posterior = model(normalized_x)
    post_samples = posterior.sample(sample_shape=torch.Size((n_posterior_samples,)))
    post_samples = [[torch.exp(log_normal_cdf(post_samples[j, i]))  for i in range(len(x))] \
        for j in range(n_posterior_samples)]
    post_samples = torch.tensor(post_samples)

    post_mean = torch.mean(post_samples, dim=0).flatten()
    post_std = torch.std(post_samples, dim=0).flatten()
    return post_mean, post_std

def evaluate_GP(model, likelihood, n_posterior_samples, x_val=None, y_val=None, n_trials_val=None, z=1.96):

    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
        
    model.eval()    
    likelihood.eval()

    with torch.no_grad():

        if x_val is None: # Poisson case-study

            n_val_points = 100
            x_val, y_val = Poisson_observations(n_val_points)
            n_trials_val=1 

        else:
            n_val_points = len(x_val)

        start = time.time()
        post_mean, post_std = posterior_predictive(model=model, x=x_val, n_posterior_samples=n_posterior_samples)
        evaluation_time = execution_time(start=start, end=time.time())

        val_satisfaction_prob = y_val.flatten()/n_trials_val
        assert val_satisfaction_prob.min()>=0
        assert val_satisfaction_prob.max()<=1

        val_dist = torch.abs(val_satisfaction_prob-post_mean)
        n_val_errors = torch.sum(val_dist > z*post_std)#/torch.sqrt(n_val_points))
        percentage_val_errors = 100*(n_val_errors/n_val_points)

        mse = torch.mean(val_dist**2)
        mre = torch.mean(val_dist/val_satisfaction_prob+0.000001)

        uncertainty_ci_area = 2*z*post_std
        avg_uncertainty_ci_area = torch.mean(uncertainty_ci_area)

        print(f"Evaluation time = {evaluation_time}")
        print(f"Mean squared error: {mse}")
        # print(f"Mean relative error: {mre}")
        print(f"Percentage of validation errors: {percentage_val_errors} %")
        print(f"Average uncertainty area:  {avg_uncertainty_ci_area}\n")

    evaluation_dict = {"percentage_val_errors":percentage_val_errors, "mse":mse, "mre":mre, 
                       "avg_uncertainty_ci_area":avg_uncertainty_ci_area, "evaluation_time":evaluation_time}

    return x_val, post_mean, post_std, evaluation_dict

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

        params_couples_idxs = list(itertools.combinations(range(len(params_list)), 2))

        fig, ax = plt.subplots(len(params_couples_idxs), 2, figsize=(9, 4*len(params_couples_idxs)))

        for row_idx, (i, j) in enumerate(params_couples_idxs):

            p1, p2 = params_list[i], params_list[j]

            axis = ax[row_idx,0] if len(params_couples_idxs)>1 else ax[0]
            data = pd.DataFrame({p1:x_val[:,i],p2:x_val[:,j],'val_counts':y_val.flatten()/n_trials_val})
            data[p1] = data[p1].apply(lambda x: format(float(x),".4f"))
            data[p2] = data[p2].apply(lambda x: format(float(x),".4f"))
            data.sort_index(level=0, ascending=True, inplace=True)

            # data = data.sort_values(by=[p1, p2], axis=0, ascending=True)
            data = data.pivot(p1, p2, "val_counts")
            sns.heatmap(data, ax=axis, label='validation pts')
            axis.set_title("validation set")

            axis = ax[row_idx,1] if len(params_couples_idxs)>1 else ax[1]

            data = pd.DataFrame({p1:x_test[:,i],p2:x_test[:,j],'posterior_preds':post_mean})
            data[p1] = data[p1].apply(lambda x: format(float(x),".4f"))
            data[p2] = data[p2].apply(lambda x: format(float(x),".4f"))
            # data = data.sort_values(by=[p1, p2], axis=0, ascending=True)
            data.sort_index(level=0, ascending=True, inplace=True)
            data = data.pivot(p1, p2, "posterior_preds")
            sns.heatmap(data, ax=axis, label='posterior preds')
            axis.set_title("posterior preds")

    plt.tight_layout()
    plt.close()
    return fig

