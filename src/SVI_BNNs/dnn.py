import os
import sys
import json
import copy
import torch
import random
import warnings
import numpy as np
from torch import nn
import torch.optim as torchopt
import torch.nn.functional as nnf

softplus = torch.nn.Softplus()
torch.set_default_dtype(torch.float32)
warnings.filterwarnings('ignore')


class DeterministicNetwork(nn.Module):
  
    def __init__(self, input_size, hidden_size, architecture_name):

        # initialize nn.Module
        super(DeterministicNetwork, self).__init__()

        output_size = 1
      
        # architecture
        if architecture_name=='2L':

            self.model = nn.Sequential(
                         nn.Linear(input_size, hidden_size),
                         nn.LeakyReLU(),
                         nn.Linear(hidden_size, output_size),
                         nn.Sigmoid()
                         )

        elif architecture_name=='3L':

            self.model = nn.Sequential(
                         nn.Linear(input_size, hidden_size),
                         nn.LeakyReLU(),
                         nn.Linear(hidden_size, hidden_size),
                         nn.LeakyReLU(),
                         nn.Linear(hidden_size, output_size),
                         nn.Sigmoid()
                         )

        self.name = "deterministic_network"
        self.architecture_name = architecture_name

    def forward(self, inputs, *args, **kwargs):
        """ Compute predictions on `inputs`. """
        return self.model(inputs)

    