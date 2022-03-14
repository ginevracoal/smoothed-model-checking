import os
import sys
import json
import copy
import random
import numpy as np

import torch
torch.set_default_dtype(torch.float32)
from torch import nn
import torch.nn.functional as nnf
softplus = torch.nn.Softplus()
import torch.optim as torchopt



import warnings
warnings.filterwarnings('ignore')


class DeterministicNetwork(nn.Module):
  
    def __init__(self, input_size, hidden_size):

        # initialize nn.Module
        super(DeterministicNetwork, self).__init__()

        output_size = 1
      
        # architecture
        self.model = nn.Sequential(
                     nn.Linear(input_size, hidden_size),
                     nn.LeakyReLU(),
                     nn.Linear(hidden_size, output_size),
                     nn.Sigmoid()
                     )
        # possibilità: togliere sigmoid da qui e metterla nel model della bnn
        self.name = "deterministic_network"

    def forward(self, inputs, *args, **kwargs):
        """ Compute predictions on `inputs`. """
        return self.model(inputs)

    
