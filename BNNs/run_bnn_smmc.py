from bnn_smmc import *

modelName = "SIR"
nb_param = 1
param_name = "Beta"

nb_obs_val = 5000
nb_config_val = 20
nb_obs_train = 10
nb_config_train = 200

prefix = "../Data/WorkingDatasets/"+modelName+"/"
df_file_val = prefix+modelName+"_DS_{}samples_{}obs_{}.pickle".format(nb_config_val, nb_obs_val, param_name)
df_file_train = prefix+modelName+"_DS_{}samples_{}obs_{}.pickle".format(nb_config_train, nb_obs_train, param_name)

bnn = BNN_smMC(df_file_train, df_file_val, nb_param, n_hidden = 50)

n_epochs = 5000
bnn.run(n_epochs = n_epochs, lr = 0.01)