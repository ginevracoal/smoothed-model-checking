import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from pcheck.semantics import stlBooleanSemantics, stlRobustSemantics
from pcheck.series.TimeSeries import TimeSeries
import pickle


class SIR_labels(object):

    def __init__(self, df_fld, df_path, param_name):
        self.modelName = 'SIR'
        self.df_fld = df_fld
        self.df_path = df_path
        self.param_name = param_name
        self.formula = '(F_[100.,120.](I<1.)) & (G_[0.,100.](I > 0))'
        self.position_dict={'S':0,'I':1,'R':2}
        self.df_np = pd.read_pickle(df_path)
        self.X = self.df_np ['X']
        self.Y = self.df_np['Y']
        self.n_params = self.Y.shape[2]
        self.VARS = ["S","I","R"]
        self.TIME = np.linspace(0.,120,241)
        self.n_configs = self.X.shape[0]
        self.n_trajs = self.X.shape[1]

    def extractVarAtPosFromX(self,fullarray, variable):
        return fullarray[:,self.position_dict[variable]]


    def extractParamsFromY(self, fullarray):
        return fullarray

    def analyze(self, experiment):
        
        S = self.extractVarAtPosFromX(experiment, 'S')
        I = self.extractVarAtPosFromX(experiment, 'I')
        R = self.extractVarAtPosFromX(experiment, 'R')
        TRAJ = np.stack([S,I,R])
        TS = TimeSeries(self.VARS, self.TIME, TRAJ)
        
        label = stlBooleanSemantics(TS, 0, self.formula)
        if label:
            return 1
        else:
            return 0

    def get_bool_labels(self):
        

        PARAMS = np.empty((self.n_configs, self.n_params))
        LABELS = np.empty((self.n_configs, self.n_trajs))

        for i in range(self.n_configs):
            PARAMS[i] = self.extractParamsFromY(self.Y[i][0])
            print("{}/{}".format(i+1,self.n_configs))
            for j in range(self.n_trajs):
                
                LABELS[i,j] = self.analyze(self.X[i][j])
                

        self.data_dict = {"params": PARAMS, "labels": LABELS}        
                
        

    def exportLabeledData(self):
        
        filename = self.df_fld+self.modelName+"_DS_{}samples_{}obs_{}.pickle".format(self.n_configs,self.n_trajs, self.param_name)
        with open(filename, 'wb') as handle:
            pickle.dump(self.data_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
  
            


if __name__ == "__main__":

    param_name = "BetaGamma"
    df_fld = "../Data/02_03_2022/SIR_22_07_11/Dataframes/"
    df_file = df_fld+"SIR_NUMPY.pkl"
    print(df_fld)
    labeler = SIR_labels(df_fld, df_file, param_name)
    labeler.get_bool_labels()
    labeler.exportLabeledData()
