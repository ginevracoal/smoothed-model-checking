import time
import torch
import gpytorch
from gpytorch.models import ApproximateGP
from gpytorch.variational import MeanFieldVariationalDistribution, CholeskyVariationalDistribution
from gpytorch.variational import VariationalStrategy, UnwhitenedVariationalStrategy


class GPmodel(ApproximateGP):

    def __init__(self, inducing_points):
        variational_distribution = CholeskyVariationalDistribution(inducing_points.size(0))
        variational_strategy = UnwhitenedVariationalStrategy(self, inducing_points, variational_distribution, 
                                                            learn_inducing_locations=False)
        super(GPmodel, self).__init__(variational_strategy)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())

    def forward(self, x):
        mean = self.mean_module(x)
        covar = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean, covar)


def execution_time(start, end):
    hours, rem = divmod(end - start, 3600)
    minutes, seconds = divmod(rem, 60)
    print("\nExecution time = {:0>2}:{:0>2}:{:0>2}".format(int(hours), int(minutes), int(seconds)))


def train_GP(model, likelihood, x_train, y_train, n_epochs, lr):

    model.train()
    likelihood.train()

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
        print(f"Epoch {i}/{n_epochs} - Loss: {loss}")

    execution_time(start=start, end=time.time())

    print("\nModel params:", model.state_dict().keys())
    return model