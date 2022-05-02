import os
import sys
import time
import pyro
import torch
import random
import scipy.io
import matplotlib
import numpy as np
from math import pi
import torch.nn as nn
from tqdm import tqdm
from pyro import poutine
import pickle5 as pickle
import torch.optim as optim
from pyro.nn import PyroModule
import matplotlib.pyplot as plt
import torch.nn.functional as F
from pyro.optim import Adam, SGD
from sklearn import preprocessing
import torch.nn.functional as nnf
from itertools import combinations
from torch.autograd import Variable
from paths import models_path, plots_path
from torch.utils.data import TensorDataset, DataLoader
from pyro.distributions import Normal, Binomial, Bernoulli
from pyro.infer import SVI, Trace_ELBO, TraceMeanField_ELBO

sys.path.append(".")
from data_utils import *
from SVI_BNNs.dnn import DeterministicNetwork
from evaluation_metrics import execution_time, evaluate_posterior_samples

softplus = torch.nn.Softplus()


class BNN_smMC(PyroModule):

    def __init__(self, model_name, list_param_names, likelihood, input_size, architecture_name, n_hidden, 
        n_test_points=20):

        # initialize PyroModule
        super(BNN_smMC, self).__init__()
        
        # BayesianNetwork extends PyroModule class
        self.det_network = DeterministicNetwork(input_size=input_size, hidden_size=n_hidden, 
            architecture_name=architecture_name)
        self.name = "bayesian_network"

        self.likelihood = likelihood
        self.input_size = input_size
        self.n_hidden = n_hidden
        self.output_size = 1
        self.n_test_points = n_test_points
        self.model_name = model_name
        self.param_name = list_param_names
        self.mre_eps = 0.000001
        self.casestudy_id = self.model_name+''.join(self.param_name)

    def model(self, x_data, y_data):

        priors = {}
    
        # set Gaussian priors on the weights of self.det_network
        for key, value in self.det_network.state_dict().items():
            loc = value #torch.zeros_like(value)
            scale = torch.ones_like(value)/value.size(dim=0)
            prior = Normal(loc=loc, scale=scale)
            priors.update({str(key):prior})

        # pyro.random_module places `priors` over the parameters of the nn.Module 
        # self.det_network and returns a distribution, which upon calling 
        # samples a new nn.Module (`lifted_module`)
        lifted_module = pyro.random_module("module", self.det_network, priors)()
    
        # samples are conditionally independent w.r.t. the observed data
        lhat = lifted_module(x_data) # out.shape = (batch_size, num_classes)
        lhat = nnf.sigmoid(lhat)

        if self.likelihood=="binomial":
            pyro.sample("obs", Binomial(total_count=self.n_trials_train, probs=lhat), obs=y_data)

        elif self.likelihood=="bernoulli":
            pyro.sample("obs", Bernoulli(probs=lhat), obs=y_data)

        else:
            raise AttributeError

    def guide(self, x_data, y_data=None):

        dists = {}
        for key, value in self.det_network.state_dict().items():

            # torch.randn_like(x) builds a random tensor whose shape equals x.shape
            loc = pyro.param(str(f"{key}_loc"), torch.randn_like(value)) 
            scale = pyro.param(str(f"{key}_scale"), torch.randn_like(value))

            # softplus is a smooth approximation to the ReLU function
            # which constraints the scale tensor to positive values
            distr = Normal(loc=loc, scale=softplus(scale))

            # add key-value pair to the samples dictionary
            dists.update({str(key):distr})
        # define a random module from the dictionary of distributions
        lifted_module = pyro.random_module("module", self.det_network, dists)()

        # compute predictions on `x_data`
        lhat = lifted_module(x_data)
        lhat = nnf.sigmoid(lhat)
        return lhat
    
    def forward(self, inputs, n_samples=100):
        """ Compute predictions on `inputs`. 
        `n_samples` is the number of samples from the posterior distribution.
        If `avg_prediction` is True, it returns the average prediction on 
        `inputs`, otherwise it returns all predictions 
        """

        preds = []
        # take multiple samples
        for i in range(n_samples):    
            pyro.set_rng_seed(i)     
            guide_trace = poutine.trace(self.guide).get_trace(inputs)
            preds.append(guide_trace.nodes['_RETURN']['value'])
        
        t_hats = torch.stack(preds).squeeze()
        t_mean = torch.mean(t_hats, 0)
        t_std = torch.std(t_hats, 0)
        
        return t_hats, t_mean, t_std
    
    def evaluate(self, train_data, val_data, n_posterior_samples, device="cpu"):

        random.seed(0)
        np.random.seed(0)
        torch.manual_seed(0)    

        if self.model_name == 'Poisson':
            raise NotImplementedError

        else:
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

            start = time.time()
            post_samples, post_mean, post_std = self.forward(x_val, n_posterior_samples)
            evaluation_time = execution_time(start=start, end=time.time())
            print(f"Evaluation time = {evaluation_time}")

        post_mean, q1, q2 , evaluation_dict = evaluate_posterior_samples(y_val=y_val,
            post_samples=post_samples, n_samples=n_samples, n_trials=n_trials)

        evaluation_dict.update({"evaluation_time":evaluation_time})
        return post_mean, q1, q2, evaluation_dict

    def train(self, train_data, n_epochs, lr, batch_size, device="cpu"):

        random.seed(0)
        np.random.seed(0)
        torch.manual_seed(0)

        if self.likelihood=='bernoulli':
            x_train, y_train, n_samples, n_trials_train = get_bernoulli_data(train_data)
            
        elif self.likelihood=='binomial':
            x_train, y_train, n_samples, n_trials_train = get_binomial_data(train_data)

        else:
            raise AttributeError

        self.n_trials_train = n_trials_train
        x_train = normalize_columns(x_train)
        y_train = y_train.unsqueeze(1)

        self.to(device)
        # x_train = x_train.to(device)
        # y_train = y_train.to(device)

        dataset = TensorDataset(x_train, y_train) 
        train_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)

        start = time.time()

        print("\nDeterministic Training:")
        self.det_network.train(train_loader=train_loader, n_trials_train=n_trials_train, epochs=500, 
            lr=0.01, likelihood=self.likelihood, device=device)

        print("\nBayesian Training:")
        # adam_params = {"lr": self.lr, "betas": (0.95, 0.999)}
        adam_params = {"lr": lr}#, "weight_decay":1.}
        optim = Adam(adam_params)
        elbo = Trace_ELBO()
        svi = SVI(self.model, self.guide, optim, loss=elbo)

        loss_history = []
        for j in tqdm(range(n_epochs)):
            loss = 0
            for x_batch, y_batch in train_loader:
                x_batch = x_batch.to(device)
                y_batch = y_batch.to(device)
                loss += svi.step(x_batch, y_batch)

            loss = loss/len(x_train)

            if (j+1)%50==0:
                print("Epoch ", j+1, "/", n_epochs, " Loss ", loss)
                loss_history.append(loss)

        training_time = execution_time(start=start, end=time.time())

        self.loss_history = loss_history
        self.n_epochs = n_epochs

        print("\nTraining time: ", training_time)
        self.training_time = training_time
        return self, training_time

    def save(self, filepath, filename, training_device):
        os.makedirs(os.path.dirname(filepath), exist_ok=True)

        param_store = pyro.get_param_store()
        print(f"\nlearned params = {param_store}")
        param_store.save(os.path.join(filepath, filename+"_"+training_device+".pt"))

        file = open(os.path.join(filepath, f"{filename}_training_time_{training_device}.txt"),"w")
        file.writelines(self.training_time)
        file.close()

        if self.n_epochs >= 50:
            fig = plt.figure()
            plt.plot(np.arange(0,self.n_epochs,50), np.array(self.loss_history))
            plt.title("loss")
            plt.xlabel("epochs")
            plt.tight_layout()
            plt.savefig(os.path.join(filepath, filename+"_loss.png"))
            plt.close()          

    def load(self, filepath, filename, training_device):

        param_store = pyro.get_param_store()
        param_store.load(os.path.join(filepath, filename+"_"+training_device+".pt"))
        for key, value in param_store.items():
            param_store.replace_param(key, value, value)

        file = open(os.path.join(filepath, f"{filename}_training_time_{training_device}.txt"),"r+")
        training_time = file.read()
        print(f"\nTraining time = {training_time}")
        return training_time