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


df = pd.read_csv("../BNNs/BNN_Learning/df.csv")
print("\n", df.head(10))

squashed_df = squash_df(df)
squashed_df['satisfaction_prob'] = squashed_df['Count_TRUE']/squashed_df['Count_ALL']
print("\n", squashed_df.head())


df_train, df_test = split_train_test(df)

### Bernoulli likelihood

# df_train['Result'] = df_train['Result'].astype(float)
# df_test['Result'] = df_test['Result'].astype(float)
# print(f"\ntraining set size = {len(df_train)}, test set size = {len(df_test)}\n")

# x_train = torch.tensor(df_train.drop('Result', axis=1).values, dtype=torch.float32)
# y_train = torch.tensor(df_train['Result'].values, dtype=torch.float32)
# train = data_utils.TensorDataset(x_train, y_train)
# train_loader = data_utils.DataLoader(train, batch_size=10, shuffle=True)

# x_test = torch.tensor(df_test.drop('Result', axis=1).values, dtype=torch.float32)
# y_test = torch.tensor(df_test['Result'].values, dtype=torch.float32)

# model = GPmodel(inducing_points=x_train)
# likelihood = gpytorch.likelihoods.BernoulliLikelihood()

# model.train()
# likelihood.train()

# optimizer = torch.optim.Adam([
#     {'params': model.parameters()},
#     {'params': likelihood.parameters()},
# ], lr=0.01)

# elbo = gpytorch.mlls.VariationalELBO(likelihood, model, num_data=y_train.size(0))

# num_epochs = 10

# for i in range(num_epochs):
#     for x_batch, y_batch in train_loader:
#         optimizer.zero_grad()
#         output = model(x_batch)
#         loss = -elbo(output, y_batch)
#         loss.backward()
#         optimizer.step()
#     print(f"Epoch {i}/{num_epochs} - Loss: {loss}")

# print("\nModel params:", model.state_dict().keys())

# os.makedirs(os.path.dirname('models/'), exist_ok=True)
# torch.save(model.state_dict(), 'models/gp_state.pth')
# state_dict = torch.load('models/gp_state.pth')
# model.load_state_dict(state_dict)

# plot_df = df_test
# plot_df['outcome'] = 'observed'

# model.eval()    
# likelihood.eval()

# with torch.no_grad():
#     observed_pred = likelihood(model(x_test))
#     pred_labels = observed_pred.mean.ge(0.5).float()

# for idx in range(len(x_test)):
#     plot_df = plot_df.append({'beta':x_test[idx, 0].item(), 'gamma':x_test[idx, 1].item(), 'Result': pred_labels[idx].item(), 
#         'outcome':'predicted'}, ignore_index=True)

# print(y_test)   
# print(pred_labels)

# path='plots/'
# os.makedirs(os.path.dirname(path), exist_ok=True)

# for param in ['beta','gamma']:

#     fig, ax = plt.subplots(1, 1, figsize=(6, 4))
#     sns.boxplot(data=plot_df, x=param, y='Result', hue='outcome')
#     fig.savefig(path+"boxplot_variational_gp_"+param+".png")
#     plt.close()

#     fig, ax = plt.subplots(1, 1, figsize=(6, 4))
#     print(plot_df.describe())
#     sns.scatterplot(data=plot_df, x=param, y='Result')
#     fig.savefig(path+"lineplot_variational_gp_"+param+".png")
#     plt.close()


### Binomial likelihood


squashed_df_train = squash_df(df_train)[['beta','gamma','Count_TRUE','Count_ALL']]
squashed_df_test = squash_df(df_test)[['beta','gamma','Count_TRUE','Count_ALL']]
squashed_df_train.rename(columns={'Count_TRUE': 'Result', 'Count_ALL':'Trials'}, inplace=True)
squashed_df_test.rename(columns={'Count_TRUE': 'Result', 'Count_ALL':'Trials'}, inplace=True)

# todo: set a fixed number of trials!
n_trials = 10

print(f"\ntraining set size = {len(squashed_df_train)}, test set size = {len(squashed_df_test)}\n")
print(squashed_df_train)

x_train = torch.tensor(squashed_df_train.drop('Result', axis=1).values, dtype=torch.float32)
y_train = torch.tensor(squashed_df_train['Result'].values, dtype=torch.float32)
train = data_utils.TensorDataset(x_train, y_train)
train_loader = data_utils.DataLoader(train, batch_size=10, shuffle=True)

x_test = torch.tensor(squashed_df_test.drop('Result', axis=1).values, dtype=torch.float32)
y_test = torch.tensor(squashed_df_test['Result'].values, dtype=torch.float32)

model = GPmodel(inducing_points=x_train)
likelihood = BinomialLikelihood(n_trials=n_trials)

model.train()
likelihood.train()

optimizer = torch.optim.Adam([
    {'params': model.parameters()},
    {'params': likelihood.parameters()},
], lr=0.01)

elbo = gpytorch.mlls.VariationalELBO(likelihood, model, num_data=y_train.size(0))

num_epochs = 10

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
torch.save(model.state_dict(), 'models/gp_state.pth')
state_dict = torch.load('models/gp_state.pth')
model.load_state_dict(state_dict)

plot_df = df_test
plot_df['outcome'] = 'observed'

model.eval()    
likelihood.eval()

with torch.no_grad():
    observed_pred = likelihood(model(x_test))
    pred_labels = observed_pred.mean.ge(0.5).float()

for idx in range(len(x_test)):
    plot_df = plot_df.append({'beta':x_test[idx, 0].item(), 'gamma':x_test[idx, 1].item(), 'Result': pred_labels[idx].item(), 
        'outcome':'predicted'}, ignore_index=True)

print(y_test)   
print(pred_labels)

path='plots/'
os.makedirs(os.path.dirname(path), exist_ok=True)

for param in ['beta','gamma']:

    fig, ax = plt.subplots(1, 1, figsize=(6, 4))
    sns.boxplot(data=plot_df, x=param, y='Result', hue='outcome')
    fig.savefig(path+"boxplot_variational_gp_"+param+".png")
    plt.close()

    fig, ax = plt.subplots(1, 1, figsize=(6, 4))
    print(plot_df.describe())
    sns.scatterplot(data=plot_df, x=param, y='Result')
    fig.savefig(path+"lineplot_variational_gp_"+param+".png")
    plt.close()
