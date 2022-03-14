from bnn_smmc import *

model_name = "Poisson"
nb_param = 1
param_name = "Lambda"

casestudy_name = model_name+param_name



if model_name == "Poisson":
	nb_obs_train = 10
	nb_config_train = 46	
	nb_obs_val = nb_obs_train
	nb_config_val = nb_config_train
else:
	nb_obs_train = 10
	nb_config_train = 200
	nb_obs_val = 5000
	nb_config_val = 20

prefix = "../Data/WorkingDatasets/"+model_name+"/"
df_file_val = prefix+model_name+"_DS_{}samples_{}obs_{}.pickle".format(nb_config_val, nb_obs_val, param_name)
df_file_train = prefix+model_name+"_DS_{}samples_{}obs_{}.pickle".format(nb_config_train, nb_obs_train, param_name)

n_hidden = 10
bnn = BNN_smMC(casestudy_name, df_file_train, df_file_val, nb_param, n_hidden)

n_epochs = 10000
bnn.run(n_epochs = n_epochs, lr = 0.01)

print(model_name, param_name, n_epochs, lr)
