import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams.update({'font.size': 22})
import pickle

df_fld = "PrGeEx/"
nb_param = 1
param_name = "k7"

'''
nb_obs_val = 5000
nb_config_val = 20

df_file_val = df_fld+"SIR_DS_{}samples_{}obs_{}.pickle".format(nb_config_val, nb_obs_val, param_name)
with open(df_file_val, 'rb') as handle:
    data_dict_val = pickle.load(handle)

params_val = data_dict_val["params"]
bools_val = data_dict_val["labels"]
#print(params_val.shape, bools_val.shape)

bool_mean_val = np.mean(bools_val, axis=1)
bool_std_val = np.std(bools_val, axis=1)
#print(bool_mean_val.shape, bool_std_val.shape)
'''
nb_obs_train = 10
nb_config_train = 25

df_file_train = df_fld+"PrGeEx_DS_{}samples_{}obs_{}.pickle".format(nb_config_train, nb_obs_train, param_name)
with open(df_file_train, 'rb') as handle:
    data_dict_train = pickle.load(handle)

params_train = data_dict_train["params"]
bools_train = data_dict_train["labels"]
print(params_train.shape, bools_train.shape)

bool_mean_train = np.mean(bools_train, axis=1)
bool_std_train = np.std(bools_train, axis=1)




fig = plt.figure()
#plt.plot(params_val.flatten(), bool_mean_val, 'b', label="valid")
#plt.fill_between(params_val.flatten(), bool_mean_val-1.96*bool_std_val/np.sqrt(nb_obs_val), bool_mean_val+1.96*bool_std_val/np.sqrt(nb_obs_val), color='b', alpha = 0.1)
plt.scatter(params_train.flatten(), bool_mean_train, marker='+',color="r",label="train")
plt.title("Satisfaction")
plt.xlabel(param_name)
plt.legend()
plt.tight_layout()
fig.savefig(df_fld+"SIR_Satisf_{}.png".format(param_name))
plt.close()
