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
from pyro import poutine
import pickle5 as pickle
import torch.optim as optim
from pyro.nn import PyroModule
import matplotlib.pyplot as plt
import torch.nn.functional as F
from pyro.optim import Adam, SGD
from sklearn import preprocessing
from itertools import combinations
from torch.autograd import Variable
from paths import models_path, plots_path
from pyro.distributions import Normal, Binomial
from pyro.infer import SVI, Trace_ELBO, TraceMeanField_ELBO

sys.path.append(".")
from BNNs.dnn import DeterministicNetwork

matplotlib.rcParams.update({'font.size': 22})
softplus = torch.nn.Softplus()

def execution_time(start, end):
    hours, rem = divmod(end - start, 3600)
    minutes, seconds = divmod(rem, 60)
    time = f"{int(hours):0>2}:{int(minutes):0>2}:{int(seconds):0>2}"
    return time


class BNN_smMC(PyroModule):

    def __init__(self, model_name, list_param_names, train_set, val_set, input_size, architecture_name='2L', 
        n_hidden=10, n_test_points=20):
        # initialize PyroModule
        super(BNN_smMC, self).__init__()
        
        # BayesianNetwork extends PyroModule class
        self.det_network = DeterministicNetwork(input_size=input_size, hidden_size=n_hidden, architecture_name=architecture_name)
        self.name = "bayesian_network"

        self.train_set_fn = train_set
        self.val_set_fn = val_set
        self.input_size = input_size
        self.n_hidden = n_hidden
        self.output_size = 1
        self.n_test_preds = n_test_points
        self.n_test_points = n_test_points
        self.model_name = model_name
        self.param_name = list_param_names
        self.mre_eps = 0.000001
        self.casestudy_id = self.model_name+''.join(self.param_name)

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
        self.MAX = xmax
        self.MIN = xmin

        self.X_train_scaled = -1+2*(self.X_train-self.MIN)/(self.MAX-self.MIN)
        self.T_train_scaled = np.expand_dims(self.T_train, axis=1)

    def load_val_data(self):
        with open(self.val_set_fn, 'rb') as handle:
            datasets_dict = pickle.load(handle)

        self.X_val = datasets_dict["params"]
        
        P_val = datasets_dict["labels"]
        n_val_points, M_val = P_val.shape

        self.M_val = M_val
        self.n_val_points = n_val_points
        self.T_val = np.sum(P_val,axis=1)
        
        self.X_val_scaled = -1+2*(self.X_val-self.MIN)/(self.MAX-self.MIN)
        
        self.T_val_scaled = self.T_val

    def model(self, x_data, y_data):

        priors = {}
    
        # set Gaussian priors on the weights of self.det_network
        for key, value in self.det_network.state_dict().items():
            loc = torch.zeros_like(value)
            scale = torch.ones_like(value)#/value.size(dim=0)
            prior = Normal(loc=loc, scale=scale)
            priors.update({str(key):prior})

        # pyro.random_module places `priors` over the parameters of the nn.Module 
        # self.det_network and returns a distribution, which upon calling 
        # samples a new nn.Module (`lifted_module`)
        lifted_module = pyro.random_module("module", self.det_network, priors)()
    
        # samples are conditionally independent w.r.t. the observed data
        lhat = lifted_module(x_data) # out.shape = (batch_size, num_classes)
        
        pyro.sample("obs", Binomial(total_count=self.M_train, probs=lhat), obs=y_data)

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
        return lhat

    
    def forward(self, inputs):
        """ Compute predictions on `inputs`. 
        `n_samples` is the number of samples from the posterior distribution.
        If `avg_prediction` is True, it returns the average prediction on 
        `inputs`, otherwise it returns all predictions 
        """

        preds = []
        # take multiple samples
        for _ in range(self.n_test_preds):         
            guide_trace = poutine.trace(self.guide).get_trace(inputs)
            preds.append(guide_trace.nodes['_RETURN']['value'])
        
        t_hats = torch.stack(preds)
        t_mean = torch.mean(t_hats, 0).numpy()
        t_std = torch.std(t_hats, 0).numpy()
        
        return preds, t_mean, t_std
    
    def set_training_options(self, n_epochs = 1000, lr = 0.01):

        self.n_epochs = n_epochs
        self.lr = lr        

    def train(self):
        random.seed(0)
        np.random.seed(0)
        torch.manual_seed(0)

        #adam_params = {"lr": self.lr, "betas": (0.95, 0.999)}
        adam_params = {"lr": self.lr}
        optim = Adam(adam_params)
        #elbo = Trace_ELBO()
        elbo = TraceMeanField_ELBO()
        svi = SVI(self.model, self.guide, optim, loss=elbo)

        batch_T_t = torch.FloatTensor(self.T_train_scaled)
        batch_X_t = torch.FloatTensor(self.X_train_scaled)

        start = time.time()

        loss_history = []
        for j in range(self.n_epochs):
            loss = svi.step(batch_X_t, batch_T_t)/ self.n_training_points
            if (j+1)%50==0:
                print("Epoch ", j+1, "/", self.n_epochs, " Loss ", loss)
                loss_history.append(loss)

        self.loss_history = loss_history

        if self.n_epochs >= 50:
            fig = plt.figure()
            plt.plot(np.arange(0,self.n_epochs,50), np.array(self.loss_history))
            plt.title("loss")
            plt.xlabel("epochs")
            plt.tight_layout()
            plt.savefig(self.plot_path+"loss.png")
            plt.close()

        training_time = execution_time(start=start, end=time.time())
        print("\nTraining time: ", training_time)
        return training_time

    def evaluate(self):
        random.seed(0)
        np.random.seed(0)
        torch.manual_seed(0)

        start = time.time()

        # it plots the histogram comparison and returns the wasserstein distance over the test set
        with torch.no_grad():

            x_val_t = torch.FloatTensor(self.X_val_scaled)

            x_test_t = []
            x_test_unscaled_t = []
            for col_idx in range(self.input_size):
                single_param_values = self.X_val_scaled[:,col_idx]
                single_param_values_unscaled = self.X_val[:,col_idx]
                x_test_t.append(torch.linspace(single_param_values.min(), single_param_values.max(), self.n_test_points))
                x_test_unscaled_t.append(torch.linspace(single_param_values_unscaled.min(), single_param_values_unscaled.max(), self.n_test_points))
            x_test_t = torch.stack(x_test_t, dim=1)

            if self.input_size>1:
                x_test_cart_t = torch.cartesian_prod(*[x_test_t[:,i] for i in range(x_test_t.shape[1])])

            x_test = x_test_t.numpy()
            x_test_unscaled = torch.stack(x_test_unscaled_t, dim=1).numpy()
            
            if self.input_size == 1:
                T_test_bnn, test_mean_pred, test_std_pred = self.forward(x_test_t)
            else: 
                T_test_bnn, test_mean_pred, test_std_pred = self.forward(x_test_cart_t)

            T_val_bnn, val_mean_pred, val_std_pred = self.forward(x_val_t)
        
        evaluation_time = execution_time(start=start, end=time.time())

        val_satisf = self.T_val_scaled/self.M_val
        val_dist = np.abs(val_satisf-val_mean_pred.flatten())
        n_val_errors = 0
        for i in range(self.n_val_points):
            if val_dist[i] > 1.96*val_std_pred[i,0]:
                n_val_errors += 1

        PercErr = 100*(n_val_errors/self.n_val_points)

        MSE = np.mean(val_dist**2)
        MRE = np.mean(val_dist/(val_satisf.flatten()+self.mre_eps))

        UncVolume = 2*1.96*test_std_pred.flatten()
        AvgUncVolume = np.mean(UncVolume)

        # if self.input_size == 1:
        #     fig = plt.figure()
        #     if self.model_name == "Poisson":
        #         poiss_satisf_fnc = lambda x: np.exp(-x)*(1+x+x**2/2+x**3/6)
        #         plt.plot(x_test, poiss_satisf_fnc(x_test_unscaled), 'b', label="valid")
        #     else: 
        #         plt.plot(self.X_val_scaled.flatten(), self.T_val_scaled/self.M_val, 'b', label="valid")
        #     plt.plot(x_test.flatten(), test_mean_pred, 'r', label="bnn")
        #     LB = test_mean_pred.flatten()-1.96*test_std_pred.flatten()
        #     plt.fill_between(x_test.flatten(), [max(lb,0) for lb in LB], test_mean_pred.flatten()+1.96*test_std_pred.flatten(), color='r', alpha = 0.1)#/np.sqrt(self.n_test_preds)
        #     plt.scatter(self.X_train_scaled.flatten(), self.T_train_scaled.flatten()/self.M_train, marker='+', color='g', label="train")
        #     plt.legend()
        #     plt.xlabel(self.param_name[0])
        #     plt.title(self.model_name)
        #     plt.tight_layout()
            
        #     figname = self.plot_path+"satisf_fnc_comparison.png"
        #     plt.savefig(figname)
        #     plt.close()

        #     fig = plt.figure()
        #     plt.plot(x_test_unscaled, UncVolume)
        #     plt.xlabel(self.param_name[0])
        #     plt.ylabel("uncertainty")
        #     plt.title(self.model_name)
        #     figname = self.plot_path+"uncertainty_volume.png"
        #     plt.tight_layout()
        #     plt.savefig(figname)
        #     plt.close()

        #     fig = plt.figure()
        #     plt.plot(self.X_val.flatten(), val_dist)
        #     plt.xlabel(self.param_name[0])
        #     plt.ylabel("absolute error")
        #     plt.title(self.model_name)
        #     figname = self.plot_path+"absolute_error.png"
        #     plt.tight_layout()
        #     plt.savefig(figname)
        #     plt.close()

        # elif self.input_size == 2:

        #     fig = plt.figure()
        #     h = plt.contourf(x_test_unscaled[:,1], x_test_unscaled[:,0], np.reshape(test_mean_pred, (self.n_test_points, self.n_test_points)))
        #     plt.colorbar()
        #     plt.xlabel(self.param_name[1])
        #     plt.ylabel(self.param_name[0])
        #     plt.title(self.model_name)
        #     plt.tight_layout()
            
        #     figname = self.plot_path+"satisf_fnc_comparison.png"
        #     plt.savefig(figname)
        #     plt.close()
            
        #     fig = plt.figure()
        #     h = plt.contourf(x_test_unscaled[:,1], x_test_unscaled[:,0], np.reshape(UncVolume, (self.n_test_points, self.n_test_points)))
        #     plt.colorbar()
        #     plt.xlabel(self.param_name[1])
        #     plt.ylabel(self.param_name[0])
        #     plt.title(self.model_name+"uncertainty")
        #     plt.tight_layout()
            
        #     figname = self.plot_path+"uncertainty.png"
        #     plt.savefig(figname)
        #     plt.close()

        #     fig = plt.figure()
        #     sqrt_val_shape = int(np.sqrt(self.n_val_points))
        #     h = plt.contourf(np.reshape(val_dist, (sqrt_val_shape, sqrt_val_shape)))
        #     plt.xlabel(self.param_name[1])
        #     plt.ylabel(self.param_name[0])
        #     plt.tight_layout()
        #     plt.colorbar()
        #     figname = self.plot_path+"absolute_error.png"
        #     plt.close()

        x_val = self.X_val_scaled
        x_val_unscaled = self.X_val

        return x_val, x_val_unscaled, val_mean_pred, val_std_pred, MSE, MRE, PercErr, AvgUncVolume, evaluation_time

    def save(self, net_name = "bnn_net.pt"):

        param_store = pyro.get_param_store()
        print(f"\nlearned params = {param_store}")
        param_store.save(self.model_path+net_name)

    def load(self, net_name = "bnn_net.pt"):
        path = self.model_path+"_"+net_name
        param_store = pyro.get_param_store()
        param_store.load(path)
        for key, value in param_store.items():
            param_store.replace_param(key, value, value)
        print("\nLoading ", path)

    def run(self, n_epochs, lr, identifier=0, train_flag=True):

        print("Loading data...")
        self.load_train_data()
        self.load_val_data()

        self.set_training_options(n_epochs, lr)

        fld_id = "epochs={}_lr={}_id={}".format(n_epochs,lr, identifier)
        self.plot_path = f"BNNs/{plots_path}/BNN_Plots_{self.casestudy_id}_{self.det_network.architecture_name}_Arch_{fld_id}/"
        self.model_path = os.path.join("BNNs",models_path,f"BNN_{self.casestudy_id}_{self.det_network.architecture_name}_Arch_{fld_id}")

        os.makedirs(self.plot_path, exist_ok=True)
        os.makedirs(f"BNNs/{models_path}", exist_ok=True)

        if train_flag:
            print("Training...")
            training_time = self.train()
            print("Saving...")
            self.save()

            file = open(os.path.join(f"BNNs/{models_path}",f"BNN_{self.casestudy_id}_{self.det_network.architecture_name}_Arch_{fld_id}"),"w")
            file.writelines(training_time)
            file.close()

        else:
            self.load()
            file = open(os.path.join(f"BNNs/{models_path}",f"BNN_{self.casestudy_id}_{self.det_network.architecture_name}_Arch_{fld_id}"),"r+")
            print(f"\nTraining time = {file.read()}")


        print("Evaluating...")
        x_test, x_test_unscaled, post_mean, post_std, mse, mre, percentage_val_errors, \
            avg_uncovered_ci_area, evaluation_time = self.evaluate()
        print("\nEvaluation time: ", evaluation_time)
        print("\nMean squared error: ", round(mse,6))
        print("Mean relative error: ", round(mre,6))
        print("Percentage of validation errors: ", round(percentage_val_errors,2), "%")
        print("Average uncertainty area: ", avg_uncovered_ci_area, "\n")

        evaluation_dict = {"percentage_val_errors":percentage_val_errors, "mse":mse, "mre":mre, 
                           "avg_uncovered_ci_area":avg_uncovered_ci_area, "evaluation_time":evaluation_time}

        return x_test_unscaled, post_mean, post_std, evaluation_dict

#todo: usare GPU