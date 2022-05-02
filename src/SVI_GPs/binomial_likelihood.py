import sys
import torch
import warnings
from gpytorch.functions import log_normal_cdf
from gpytorch.distributions import base_distributions

sys.path.append(".")
from SVI_GPs.one_dimensional_likelihood import _OneDimensionalLikelihood


class BinomialLikelihood(_OneDimensionalLikelihood):

    def __init__(self):
        super(BinomialLikelihood, self).__init__()

    def forward(self, function_samples, **kwargs):
        # conditional distribution p(y|f(x))
        output_probs = torch.tensor(base_distributions.Normal(0, 1).cdf(function_samples))
        return base_distributions.Binomial(total_count=self.n_trials, probs=output_probs)

    def log_marginal(self, observations, function_dist, *args, **kwargs):
        marginal = self.marginal(function_dist, *args, **kwargs)
        return marginal.log_prob(observations)

    def marginal(self, function_dist, **kwargs):
        # predictive distribution
        mean = function_dist.mean
        var = function_dist.variance
        link = mean.div(torch.sqrt(1 + var))
        output_probs = base_distributions.Normal(0, 1).cdf(link)
        return base_distributions.Binomial(total_count=self.n_trials, probs=output_probs)

    def expected_log_prob(self, observations, function_dist, *params, **kwargs):
        flat_obs = observations.flatten()
        n = torch.tensor([self.n_trials for _ in range(len(flat_obs))], dtype=torch.float32).to(flat_obs.device)

        # expected log likelihood over the variational GP distribution

        def log_prob_lambda(function_samples):
            log_bin_coeff = torch.lgamma(n + 1) - torch.lgamma((n - flat_obs) + 1) - torch.lgamma(flat_obs + 1)
            second_log_trm = observations.mul(log_normal_cdf(function_samples))+(self.n_trials-observations).mul(log_normal_cdf(-function_samples))
            return log_bin_coeff+second_log_trm

        log_prob = self.quadrature(log_prob_lambda, function_dist)
        return log_prob

