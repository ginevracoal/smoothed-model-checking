#!/usr/bin/env python3

import sys
import math
import torch
import warnings
from copy import deepcopy
from abc import ABC, abstractmethod
from gpytorch.likelihoods.likelihood import Likelihood

sys.path.append(".")
from SVI_GPs.quadrature import GaussHermiteQuadrature1D


class _OneDimensionalLikelihood(Likelihood, ABC):
    r"""
    A specific case of :obj:`~gpytorch.likelihoods.Likelihood` when the GP represents a one-dimensional
    output. (I.e. for a specific :math:`\mathbf x`, :math:`f(\mathbf x) \in \mathbb{R}`.)
    Inheriting from this likelihood reduces the variance when computing approximate GP objective functions
    by using 1D Gauss-Hermite quadrature.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.quadrature = GaussHermiteQuadrature1D()

    def expected_log_prob(self, observations, function_dist, *args, **kwargs):
        log_prob_lambda = lambda function_samples: self.forward(function_samples).log_prob(observations)
        log_prob = self.quadrature(log_prob_lambda, function_dist)
        return log_prob

    def log_marginal(self, observations, function_dist, *args, **kwargs):
        prob_lambda = lambda function_samples: self.forward(function_samples).log_prob(observations).exp()
        prob = self.quadrature(prob_lambda, function_dist)
        return prob.log()