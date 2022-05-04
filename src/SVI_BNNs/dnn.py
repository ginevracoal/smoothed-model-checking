import os
import sys
import json
import copy
import time
import torch
import random
import warnings
import numpy as np
from torch import nn
import torch.optim as torchopt
import torch.nn.functional as nnf
from torch.distributions import Binomial, Bernoulli

from data_utils import *

softplus = torch.nn.Softplus()
torch.set_default_dtype(torch.float32)
warnings.filterwarnings('ignore')


class DeterministicNetwork(nn.Module):
  
    def __init__(self, input_size, hidden_size, architecture_name, activation_function='leaky'):

        # initialize nn.Module
        super(DeterministicNetwork, self).__init__()
        output_size = 1

        if activation_function=='leaky':
            activation = nn.LeakyReLU

        elif activation_function=='tanh':
            activation = nn.Tanh

        if architecture_name=='2L':

            self.model = nn.Sequential(
                         nn.Flatten(),
                         nn.Linear(input_size, hidden_size),
                         activation(),
                         nn.Linear(hidden_size, output_size),
                         # nn.Sigmoid()
                         )

        elif architecture_name=='3L':

            self.model = nn.Sequential(
                         nn.Flatten(),
                         nn.Linear(input_size, hidden_size),
                         activation(),
                         nn.Linear(hidden_size, hidden_size),
                         activation(),
                         nn.Linear(hidden_size, output_size),
                         # nn.Sigmoid()
                         )

        self.name = "deterministic_network"
        self.architecture_name = architecture_name

    def forward(self, inputs, *args, **kwargs):
        """ Compute predictions on `inputs`. """
        return self.model(inputs)

    def loss_func(self, likelihood, probs, n_trials_train, y_batch):

        if likelihood=="binomial":
            dist = Binomial(total_count=n_trials_train, probs=probs)

        elif likelihood=="bernoulli":
            dist = Bernoulli(probs=probs)

        else:
            raise AttributeError

        return -dist.log_prob(y_batch).sum(0)

    def train(self, train_loader, n_trials_train, epochs, lr, likelihood, device="cpu"):

        random.seed(0)
        np.random.seed(0)
        torch.manual_seed(0)

        self.to(device)
        optimizer = torchopt.Adam(params=self.parameters(), lr=lr)

        for epoch in range(epochs):
            total_loss = 0.0
            correct_predictions = 0.0

            for x_batch, y_batch in train_loader:
                x_batch = x_batch.to(device)
                y_batch = y_batch.to(device)

                optimizer.zero_grad()
                outputs = self.forward(x_batch, device)
                outputs = nnf.sigmoid(outputs)
                loss = self.loss_func(likelihood=likelihood, probs=outputs, n_trials_train=n_trials_train, 
                    y_batch=y_batch)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            total_loss = total_loss / len(train_loader.dataset)
            
            if (epoch+1)%50==0:
                print("Epoch ", epoch+1, "/", epochs, " Loss ", total_loss)


    def evaluate(self, train_data, val_data, device="cpu"):

        random.seed(0)
        np.random.seed(0)
        torch.manual_seed(0)    

        x_train = get_binomial_data(train_data)[0]
        x_val, _, n_samples, n_trials = get_tensor_data(val_data)

        min_x, max_x, _ = normalize_columns(x_train, return_minmax=True)
        x_val = normalize_columns(x_val, min_x=min_x, max_x=max_x) 
        y_val = torch.tensor(val_data["labels"], dtype=torch.float32)

        self.to(device)
        x_train = x_train.to(device)
        x_val = x_val.to(device)
        y_val = y_val.to(device)

        with torch.no_grad():   
            out = self.forward(x_val)
            out = nnf.sigmoid(out)
        return out.squeeze()
