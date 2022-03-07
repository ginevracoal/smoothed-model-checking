import torch
import warnings
from gpytorch.functions import log_normal_cdf
from gpytorch.distributions import base_distributions
from gpytorch.likelihoods.likelihood import _OneDimensionalLikelihood


class BinomialLikelihood(_OneDimensionalLikelihood):

    def __init__(self, n_trials):
        super(BinomialLikelihood, self).__init__()
        self.n_trials = n_trials

    def forward(self, function_samples, **kwargs):
        # conditional distribution p(y|f(x))
        output_probs = base_distributions.Normal(0, 1).cdf(function_samples)
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
        # expected log likelihood over the variational GP distribution
        log_prob_lambda = lambda function_samples: log_normal_cdf(function_samples.mul(observations))
        log_prob = self.quadrature(log_prob_lambda, function_dist)
        return log_prob