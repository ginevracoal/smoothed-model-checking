import torch
import gpytorch
from gpytorch.models import ApproximateGP
from gpytorch.variational import VariationalStrategy
from gpytorch.variational import MeanFieldVariationalDistribution


class GPmodel(ApproximateGP):

    def __init__(self, inducing_points):
        variational_distribution = MeanFieldVariationalDistribution(inducing_points.size(0))
        variational_strategy = VariationalStrategy(self, inducing_points, variational_distribution, 
                                                    learn_inducing_locations=False)
        super(GPmodel, self).__init__(variational_strategy)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())

    def forward(self, x):
        mean = self.mean_module(x)
        covar = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean, covar)



def train_GP(model, likelihood, x_train, y_train, num_epochs):

    model.train()
    likelihood.train()

    optimizer = torch.optim.Adam([{'params': model.parameters()}, {'params': likelihood.parameters()}], lr=0.01)
    elbo = gpytorch.mlls.VariationalELBO(likelihood, model, num_data=len(x_train))

    print()
    for i in range(num_epochs):
        optimizer.zero_grad()
        output = model(x_train)
        loss = -elbo(output, y_train)
        loss.backward()
        optimizer.step()
        print(f"Epoch {i}/{num_epochs} - Loss: {loss}")

    print("\nModel params:", model.state_dict().keys())
    return model