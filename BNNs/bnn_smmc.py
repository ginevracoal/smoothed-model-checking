import torch
import torch.nn as nn
import torch.optim as optim

import numpy as np
from sklearn.datasets import make_moons
from sklearn import preprocessing

import matplotlib.pyplot as plt

import scipy.io
import pyro
from pyro.infer import SVI, Trace_ELBO, TraceMeanField_ELBO
from pyro.optim import Adam, SGD
import torch.nn.functional as F
from pyro.distributions import Normal, Binomial
from math import pi
import pickle
import time
import os
from pyro import poutine

from torch.autograd import Variable
from itertools import combinations

from pyro.nn import PyroModule
from DNN import DeterministicNetwork

softplus = torch.nn.Softplus()

class BNN_smMC(PyroModule):

    def __init__(self, train_set, val_set, input_size, n_hidden = 10):
        # initialize PyroModule
        super(BNN_smMC, self).__init__()
        
        # BayesianNetwork extends PyroModule class
        self.det_network = DeterministicNetwork(input_size, n_hidden)
        self.name = "bayesian_network"

        self.train_set_fn = train_set
        self.val_set_fn = val_set
        self.input_size = input_size
        self.n_hidden = n_hidden
        self.output_size = 1
        self.n_val_preds = 100

    def load_train_data(self):
        with open(self.train_set_fn, 'rb') as handle:
            datasets_dict = pickle.load(handle)

        self.X_train = datasets_dict["params"]
        
        P_train = datasets_dict["labels"]
        n_train_points, M_train = P_train.shape

        self.M_train = M_train
        self.n_training_points = n_train_points
        self.T_train = np.sum(P_train,axis=1)

        xmax = np.max(self.X_train, axis = 0)
        xmin = np.min(self.X_train, axis = 0)
        #tmax = np.max(np.max(self.T_train, axis = 0), axis = 0)
        #tmin = np.min(np.min(self.T_train, axis = 0), axis = 0)
        self.MAX = xmax
        self.MIN = xmin

        self.X_train_scaled = -1+2*(self.X_train-self.MIN)/(self.MAX-self.MIN)
        #self.T_train_scaled = -1+2*(self.T_train-self.MIN[1])/(self.MAX[1]-self.MIN[1])
        #self.X_train_scaled = self.X_train
        self.T_train_scaled = self.T_train

    def load_val_data(self):
        with open(self.val_set_fn, 'rb') as handle:
            datasets_dict = pickle.load(handle)

        self.X_val = datasets_dict["params"]
        
        P_val = datasets_dict["labels"]
        n_val_points, M_val = P_val.shape

        self.M_val = M_val
        self.n_val_points = n_val_points
        self.T_val = np.sum(P_val,axis=1)
        
        #self.X_val_scaled = self.X_val
        self.X_val_scaled = -1+2*(self.X_val-self.MIN)/(self.MAX-self.MIN)
        
        self.T_val_scaled = self.T_val

    def model(self, x_data, y_data):

        priors = {}
    
        # set Gaussian priors on the weights of self.det_network
        for key, value in self.det_network.state_dict().items():
            loc = torch.zeros_like(value)
            scale = torch.ones_like(value)
            prior = Normal(loc=loc, scale=scale)
            priors.update({str(key):prior})

        # pyro.random_module places `priors` over the parameters of the nn.Module 
        # self.det_network and returns a distribution, which upon calling 
        # samples a new nn.Module (`lifted_module`)
        lifted_module = pyro.random_module("module", self.det_network, priors)()
    

        # samples are conditionally independent w.r.t. the observed data
        with pyro.plate("data", len(x_data)):

            lhat = lifted_module(x_data) # out.shape = (batch_size, num_classes)
        
            obs = pyro.sample("obs", Binomial(total_count=self.M_train, probs=lhat), obs=y_data)
            

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

        with pyro.plate("data", len(x_data)):
            
            #************************************* PERPLESSITA' QUI *************************************
            # compute predictions on `x_data`
            lhat = lifted_module(x_data)
            
        return lhat

    def forward(self, inputs):
        """ Compute predictions on `inputs`. 
        `n_samples` is the number of samples from the posterior distribution.
        If `avg_prediction` is True, it returns the average prediction on 
        `inputs`, otherwise it returns all predictions 
        """
        

        preds = []
        # take multiple samples
        for _ in range(self.n_val_preds):         
            guide_trace = poutine.trace(self.guide).get_trace(inputs)  
            preds.append(guide_trace.nodes['_RETURN']['value'])
        
        t_hats = torch.stack(preds)
        t_mean = torch.mean(t_hats, 0).numpy()
        t_std = torch.std(t_hats, 0).numpy()

        return preds, t_mean, t_std




    def set_training_options(self, n_epochs = 1000, lr = 0.01):

        self.n_epochs = n_epochs
        self.lr = lr
        #self.net = NN(self.input_size,self.n_hidden,self.output_size)


    def train(self):

        #adam_params = {"lr": self.lr}
        adam_params = {"lr": self.lr, "betas": (0.95, 0.999)}
        optim = Adam(adam_params)
        elbo = Trace_ELBO()
        svi = SVI(self.model, self.guide, optim, loss=elbo)

        batch_T_t = torch.FloatTensor(self.T_train_scaled)
        batch_X_t = torch.FloatTensor(self.X_train_scaled)

        start_time = time.time()

        loss_history = []
        for j in range(self.n_epochs):
            loss = svi.step(batch_X_t, batch_T_t)/ self.n_training_points
            if (j+1)%50==0:
                print("Epoch ", j+1, "/", self.n_epochs, " Loss ", loss)
                loss_history.append(loss)

        self.loss_history = loss_history


    def evaluate(self, iter_id = None, fld_id = None):

        # it prints the histogram comparison and returns the wasserstein distance over the test set

        X_val_t = torch.FloatTensor(self.X_val_scaled)

        with torch.no_grad():

            T_val_bnn, val_mean_pred, val_std_pred = self.forward(X_val_t)
        
        MSE = np.mean((self.T_val_scaled-val_mean_pred)**2)

        plot_path = "BNN_Plots"

        if iter_id:
            BNN_PP = plot_path+"/3L_Arch_{}/".format(fld_id)
        else:
            BNN_PP = plot_path+"/3L_Arch_{}/".format(fld_id)
        os.makedirs(BNN_PP, exist_ok=True)

        self.results_path = BNN_PP
    
        fig = plt.figure()
        plt.plot(np.arange(int(self.n_epochs/50)), np.array(self.loss_history))
        plt.title("loss")
        plt.savefig(BNN_PP+"loss.png")

        if self.input_size == 1:
            fig = plt.figure()
            plt.plot(self.X_val_scaled.flatten(), self.T_val_scaled/self.M_val, 'b', label="valid")
            plt.plot(self.X_val_scaled.flatten(), val_mean_pred, 'r', label="bnn")
            plt.fill_between(self.X_val_scaled.flatten(), val_mean_pred.flatten()-1.96*val_std_pred.flatten()/np.sqrt(self.n_val_preds), val_mean_pred.flatten()+1.96*val_std_pred.flatten()/np.sqrt(self.n_val_preds), color='r', alpha = 0.1)
            plt.legend()
            
            
            figname = BNN_PP+"satisf_fnc_comparison.png"
            plt.savefig(figname)
            plt.close()

        return MSE






    def save(self, net_name = "bnn_net.pt"):

        param_store = pyro.get_param_store()
        print(f"\nlearned params = {param_store}")
        param_store.save(self.results_path+net_name)


    def run(self, n_epochs = 100, lr = 0.01):

        print("Loading data...")
        self.load_train_data()
        self.load_val_data()

        self.set_training_options(n_epochs, lr)

        fld_id = "epochs={}_lr={}_id={}".format(n_epochs,lr,time.time())
        print("Training...")
        self.train()
        
        print("Evaluating...")
        mse = self.evaluate(fld_id = fld_id)
        print("Mean squared error: ", mse)

        print("Saving...")
        self.save()