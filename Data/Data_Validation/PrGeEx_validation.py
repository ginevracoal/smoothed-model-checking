import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from pcheck.semantics import stlBooleanSemantics, stlRobustSemantics
from pcheck.series.TimeSeries import TimeSeries
import pickle
import os
import matplotlib
matplotlib.rcParams.update({'font.size': 22})

class PrGeEx_labels(object):

    def __init__(self, df_fld, df_path, mean_df_fld, mean_df_path, param_name):
        self.modelName = 'PrGeEx'
        self.df_fld = df_fld
        self.df_path = df_path
        self.mean_df_fld = mean_df_fld
        self.mean_df_path = mean_df_path
        self.param_name = param_name
        self.formula = 'F_[0,21000](G_[0,5000]((R<0)&(L>0)))'
        self.position_dict={'PLac':0, 'PLac':1, 'PLacRNAP':2, 'TrLacZ1':3, 'RbsLacZ':4, 'TrLacZ2':5, 'Ribosome':6, 'RbsRibosome':7, 'TrRbsLacZ':8, 'LacZ':9, 'dgrLacZ':10, 'dgrRbsLacZ':11}
        self.df_np = pd.read_pickle(df_path)
        self.mean_df_np = pd.read_pickle(mean_df_path)
        self.X = self.df_np ['X']
        self.X_mean = self.mean_df_np['X']
        self.Y = self.df_np['Y']
        self.n_params = self.Y.shape[2]
        self.VARS = ['L', 'R']
        self.TIME = np.linspace(0,21000,4200)
        self.n_configs = self.X.shape[0]
        self.n_trajs = self.X.shape[1]
        self.n_trajs_mean = self.X_mean.shape[1]
        self.n_steps = 4200

    def extractVarAtPosFromX(self,fullarray, variable):
        return fullarray[:,self.position_dict[variable]]


    def extractParamsFromY(self, fullarray):
        return fullarray

    def analyze(self, experiment, mean_LacZ):
        
        LacZ = self.extractVarAtPosFromX(experiment, 'LacZ')
        LacZ = LacZ[:self.n_steps]

        LeftSide = LacZ-mean_LacZ+0.1*mean_LacZ
        RightSide = LacZ-mean_LacZ-0.1*mean_LacZ

        TRAJ = np.stack([LeftSide, RightSide])
        TS = TimeSeries(self.VARS, self.TIME, TRAJ)
        
        label = stlBooleanSemantics(TS, 0, self.formula)
        if label:
            return 1
        else:
            return 0

    def plot_trajs(self):
        
        os.makedirs(self.df_fld+"plots_w_avg/", exist_ok = True)
        for i in range(self.n_configs):
            fig = plt.figure()
            for j in range(self.n_trajs):
                LacZ_ij = self.extractVarAtPosFromX(self.X[i][j], 'LacZ')       
                LacZ_ij = LacZ_ij[:self.n_steps]
                plt.plot(self.TIME, LacZ_ij, color='b')
            plt.plot(self.TIME, self.AvgLacZ[i], color='r')    
            plt.xlabel("time")
            plt.ylabel("LacZ")
            plt.title(self.param_name+"={}".format(self.Y[i][0]))
            plt.tight_layout()
            fig.savefig(self.df_fld+"plots_w_avg/LacZ_traj_{}_{}.png".format(i, self.param_name))
            plt.close()


    def compute_avg_trajs(self):

        AvgLacZ = np.zeros((self.n_configs, self.n_steps))
        for i in range(self.n_configs):
            for j in range(self.n_trajs_mean):
                AvgLacZ[i] += self.extractVarAtPosFromX(self.X_mean[i][j], 'LacZ')/self.n_trajs_mean

        self.AvgLacZ = AvgLacZ




    def get_bool_labels(self):
        
        PARAMS = np.empty((self.n_configs, self.n_params))
        LABELS = np.empty((self.n_configs, self.n_trajs))

        for i in range(self.n_configs):
            PARAMS[i] = self.extractParamsFromY(self.Y[i][0])

            print("{}/{}".format(i+1,self.n_configs))
            for j in range(self.n_trajs):
                
                LABELS[i,j] = self.analyze(self.X[i][j], self.AvgLacZ[i])
                

        self.data_dict = {"params": PARAMS, "labels": LABELS}        
                
        

    def exportLabeledData(self):
        
        filename = self.df_fld+self.modelName+"_DS_{}samples_{}obs_{}.pickle".format(self.n_configs,self.n_trajs, self.param_name)
        with open(filename, 'wb') as handle:
            pickle.dump(self.data_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
  
            


if __name__ == "__main__":

    param_name = "k2k7"
    df_fld = "../Data/04_03_2022/PrGeEx_12_31_56/Dataframes/"
    df_file = df_fld+"PrGeEx_NUMPY.pkl"
    mean_df_fld = "../Data/04_03_2022/PrGeEx_12_31_56/Dataframes/"
    mean_df_file = mean_df_fld+"PrGeEx_NUMPY.pkl"
    print(param_name, df_fld, mean_df_fld)
    labeler = PrGeEx_labels(df_fld, df_file, mean_df_fld, mean_df_file, param_name)
    labeler.compute_avg_trajs()
    #labeler.plot_trajs()
    print(labeler.X.shape,labeler.X_mean.shape)
    labeler.get_bool_labels()
    labeler.exportLabeledData()
