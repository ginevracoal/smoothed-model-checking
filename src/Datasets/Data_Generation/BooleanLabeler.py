import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from pcheck.semantics import stlBooleanSemantics, stlRobustSemantics
from pcheck.series.TimeSeries import TimeSeries
import pickle
from pathlib import Path
from tqdm import tqdm

class BooleanLabeler(object):


    def __init__(self, args_dict):
        self.D = args_dict
        self.timeline = np.linspace(0,self.D["T"], self.D["n_steps"])
        self.df_path = self.D["dataFolder"] / Path("WorkingDatasets/"+self.D["modelName"]+"/")
    
    def set_traj_dict(self):
        self.X = self.D["traj_dict"]["X"] # trajectories
        self.Y = self.D["traj_dict"]["Y"] # parameters

    def extractVarAtPosFromX(self,fullarray, variable):
        return fullarray[:,self.D["position_dict"][variable]]


    def extractParamsFromY(self, fullarray):
        return fullarray

    def compute_avg_traj(self): 
        # method needed and valid only for PrGeEx
        AvgLacZ = np.zeros((self.D["n_combination"], self.D["n_steps"]))
        for i in range(self.D["n_combination"]):
            for j in range(int(self.D["n_trajs_mean"])):
                AvgLacZ[i] += self.extractVarAtPosFromX(self.D["mean_data_dict"]["X"][i][j], 'LacZ')/self.D["n_trajs_mean"]
        self.AvgLacZ = AvgLacZ


    def analyze(self, experiment, mean_experiment=None):

        if self.D["modelName"] == "PrGeEx":
    
            LacZ = self.extractVarAtPosFromX(experiment, 'LacZ')

            LeftSide = LacZ-mean_experiment+0.1*mean_experiment
            RightSide = LacZ-mean_experiment-0.1*mean_experiment

            trajectories = np.stack([LeftSide, RightSide])
            formula_variables = ['L', 'R']

        else:
            trajectories = np.stack([self.extractVarAtPosFromX(experiment, species) for species in self.D["variables"]])
            formula_variables = self.D["variables"]
        
        time_series = TimeSeries(formula_variables, self.timeline, trajectories)
                
        label = stlBooleanSemantics(time_series, 0, self.D["formula"])
        if label:
            return 1
        else:
            return 0



    def get_bool_labels(self):
        

        parameters = np.empty((self.D["n_combination"], self.D["param_space_dim"]))
        labels = np.empty((self.D["n_combination"], self.D["n_trajs"]))

        if self.D["modelName"] == "PrGeEx":
            self.compute_avg_traj()

        for i in tqdm(range(self.D["n_combination"])):

            parameters[i] = self.extractParamsFromY(self.Y[i][0])        
            
            for j in range(self.D["n_trajs"]):
                
                if self.D["modelName"] == "PrGeEx":
                    labels[i,j] = self.analyze(self.X[i][j], self.AvgLacZ[i])
                else:
                    labels[i,j] = self.analyze(self.X[i][j])
                

        self.D["dataset_dict"] = {"params": parameters, "labels": labels}


    def exportLabeledData(self):
        
        if self.D["latinFlag"]:
            filename = self.df_path / Path(self.D["modelName"]+"_DS_{}_latin_samples_{}obs_{}.pickle".format(self.D["n_combination"],self.D["n_trajs"], ''.join(self.D["params"])) )
        else:
            filename = self.df_path / Path(self.D["modelName"]+"_DS_{}samples_{}obs_{}.pickle".format(self.D["n_combination"],self.D["n_trajs"], ''.join(self.D["params"])) )

        with open(filename, 'wb') as handle:
            pickle.dump(self.D["dataset_dict"], handle, protocol=pickle.HIGHEST_PROTOCOL)
          