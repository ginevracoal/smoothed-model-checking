import sys
import torch
import warnings
from gpytorch.functions import log_normal_cdf
from gpytorch.distributions import base_distributions

sys.path.append(".")
from SVI_GPs.one_dimensional_likelihood import _OneDimensionalLikelihood


class BernoulliLikelihood(_OneDimensionalLikelihood):

    def __init__(self):
        super(BernoulliLikelihood, self).__init__()

    def forward(self, function_samples, **kwargs):
        # conditional distribution p(y|f(x))
        output_probs = torch.tensor(base_distributions.Normal(0, 1).cdf(function_samples))
        return base_distributions.Bernoulli(probs=output_probs)

    def log_marginal(self, observations, function_dist, *args, **kwargs):
        marginal = self.marginal(function_dist, *args, **kwargs)
        return marginal.log_prob(observations)

    def marginal(self, function_dist, **kwargs):
        # predictive distribution
        mean = function_dist.mean
        var = function_dist.variance
        link = mean.div(torch.sqrt(1 + var))
        output_probs = base_distributions.Normal(0, 1).cdf(link)
        return base_distributions.Bernoulli(probs=output_probs)

    def expected_log_prob(self, observations, function_dist, *params, **kwargs):
        if torch.any(observations.eq(-1)):
            # Remove after 1.0
            warnings.warn(
                "BernoulliLikelihood.expected_log_prob expects observations with labels in {0, 1}. "
                "Observations with labels in {-1, 1} are deprecated.",
                DeprecationWarning,
            )
        else:
            observations = observations.mul(2).sub(1)

        # expected log likelihood over the variational GP distribution

        log_prob_lambda = lambda function_samples: log_normal_cdf(function_samples.mul(observations))
        log_prob = self.quadrature(log_prob_lambda, function_dist)
        return log_prob