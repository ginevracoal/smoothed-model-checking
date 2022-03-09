import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams.update({'font.size': 22})
import pickle
import os

df_fld = "PhosRelay/"
nb_param = 1
param_name = "k1"


ds_id = "14_54_42"
df_file_train = "../08_03_2022/PhosRelay_{}/Dataframes/PhosRelay_NUMPY.pkl".format(ds_id)

pl_id = "plots_{}/".format(ds_id)
os.makedirs(df_fld+pl_id, exist_ok=True)

with open(df_file_train, 'rb') as handle:
    data_train_dict = pickle.load(handle)

X = data_train_dict["X"]
Y = data_train_dict["Y"]
n_config, n_trajs, n_steps, n_species = X.shape
timeline = np.linspace(0,1200, n_steps)


if True:
    for i in range(n_config):
        print(i)
        fig = plt.figure()
        for j in range(n_trajs):
            #plt.plot(timeline, X[i,j,:,0], 'k')
            plt.plot(timeline, X[i,j,:,2], 'b')
            plt.plot(timeline, X[i,j,:,4], 'g')
            plt.plot(timeline, X[i,j,:,6], 'r')
        plt.xlabel("time")
        plt.tight_layout()
        fig.savefig(df_fld+pl_id+"/PhosRelay_Trajs_{}_Point{}.png".format(param_name, i))
        plt.close()
