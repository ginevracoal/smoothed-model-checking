import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from pcheck.semantics import stlBooleanSemantics, stlRobustSemantics
from pcheck.series.TimeSeries import TimeSeries
import pickle


class PhosRelay_labels(object):

    def __init__(self, df_fld, df_path, param_name):
        self.modelName = 'PhosRelay'
        self.df_fld = df_fld
        self.df_path = df_path
        self.param_name = param_name
        self.formula = '(G_[0.,200.](D13 >= 0)) & (G_[600.,1200.](D13 < 0))'
        self.position_dict={'B':0,'L1':1,'L1p':2,'L2':3,'L2p':4,'L3':5,'L3p':6}
        self.df_np = pd.read_pickle(df_path)
        self.X = self.df_np ['X']
        self.Y = self.df_np['Y']
        self.n_params = self.Y.shape[2]
        self.VARS = ["D13"]
        self.TIME = np.linspace(0.,120,241)
        self.n_configs = self.X.shape[0]
        self.n_trajs = self.X.shape[1]

    def extractVarAtPosFromX(self,fullarray, variable):
        return fullarray[:,self.position_dict[variable]]


    def extractParamsFromY(self, fullarray):
        return fullarray

    def analyze(self, experiment):
        
        L1p = self.extractVarAtPosFromX(experiment, 'L1p')
        L3p = self.extractVarAtPosFromX(experiment, 'L3p')
        D13 = L1p-L3p
        TRAJ = np.stack([D13])
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

    param_name = "k1k2k3"
    df_fld = "../Data/09_03_2022/PhosRelay_16_40_17/Dataframes/"
    df_file = df_fld+"PhosRelay_NUMPY.pkl"
    print(df_fld)
    labeler = PhosRelay_labels(df_fld, df_file, param_name)
    labeler.get_bool_labels()
    labeler.exportLabeledData()
