import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams.update({'font.size': 22})
import pickle

nb_obs = 10
nb_param = 1
nb_config = 46

df_fld = "Poisson/"
df_file = df_fld+"Poisson_DS_{}samples_{}obs_Lambda.pickle".format(nb_config, nb_obs)

with open(df_file, 'rb') as handle:
    data_dict = pickle.load(handle)

params = data_dict["params"]
bools = data_dict["labels"]

bool_mean = np.mean(bools, axis=1)
bool_std = np.std(bools, axis=1)


satisf_fnc = lambda x: np.exp(-x)*(1+x+x**2/2+x**3/6)
satisf = satisf_fnc(params.flatten())
fig = plt.figure()
plt.plot(params.flatten(), satisf,color='b', label="exact")
plt.scatter(params.flatten(), bool_mean, marker='+',color="r",label="train")
plt.title("Satisfaction")
plt.xlabel("$\lambda$")
plt.legend()
plt.tight_layout()
fig.savefig(df_fld+"Poisson_Satisf_{}samples_{}obs_lam.png".format(nb_config,nb_obs))
plt.close()