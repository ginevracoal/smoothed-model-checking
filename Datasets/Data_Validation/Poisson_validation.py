import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from pcheck.semantics import stlBooleanSemantics, stlRobustSemantics
from pcheck.series.TimeSeries import TimeSeries
import pickle


class Poisson_labels(object):

    def __init__(self, df_fld, df_path, param_name):
        self.modelName = 'Poisson'
        self.df_fld = df_fld
        self.df_path = df_path
        self.param_name = param_name
        self.formula = 'G_[0,1] (N < 4)'
        self.position_dict={'N':0}
        self.df_np = pd.read_pickle(df_path)
        self.X = self.df_np['X']
        self.Y = self.df_np['Y']
        self.n_params = self.Y.shape[2]
        self.VARS = ['N']
        self.TIME = np.linspace(0.1,1.,10)
        self.n_configs = self.X.shape[0]
        self.n_trajs = self.X.shape[1]

    def extractVarAtPosFromX(self,fullarray, variable):
        return fullarray[:,self.position_dict[variable]]


    def extractParamsFromY(self, fullarray):
        return fullarray

    def analyze(self, experiment):

        N = self.extractVarAtPosFromX(experiment, 'N')

        TRAJ = np.stack([N])
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
            
            for j in range(self.n_trajs):
                
                LABELS[i,j] = self.analyze(self.X[i][j])
                

        self.data_dict = {"params": PARAMS, "labels": LABELS}
                
                



    def exportLabeledData(self):
        
        filename = self.df_fld+self.modelName+"_DS_{}samples_{}obs_{}.pickle".format(self.n_configs,self.n_trajs, self.param_name)
        with open(filename, 'wb') as handle:
            pickle.dump(self.data_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
           


if __name__ == "__main__":

    param_name = "Lambda"
    df_fld = "../Data/02_03_2022/Poisson_21_44_15/Dataframes/"
    df_file = df_fld+"Poisson_NUMPY.pkl"
    print(df_fld)
    labeler = Poisson_labels(df_fld, df_file, param_name)

    labeler.get_bool_labels()
    labeler.exportLabeledData()  
