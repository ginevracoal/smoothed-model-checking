import os, sys
from smMC_GPEP import *
import pickle
import numpy as np

modelName = "SIR"
params = ["beta"]
print(modelName, params)
if len(params) == 1:
	paramterName = params[0]
	train_path = "../../Data/WorkingDatasets/{}/{}_DS_500samples_50obs_{}.pickle".format(modelName,modelName,paramterName)    
	test_path = "../../Data/WorkingDatasets/{}/{}_DS_1000samples_1000obs_{}.pickle".format(modelName,modelName,paramterName)
else:
	paramterName = ''.join(params)
	train_path = "../../Data/WorkingDatasets/{}/{}_DS_2500samples_50obs_{}.pickle".format(modelName,modelName,paramterName)    
	test_path = "../../Data/WorkingDatasets/{}/{}_DS_400samples_1000obs_{}.pickle".format(modelName,modelName,paramterName)

with open(train_path, 'rb') as handle:
	trainsets_dict = pickle.load(handle)

X_train = trainsets_dict["params"]
P_train = trainsets_dict["labels"]       
n_train_points, M_train = P_train.shape
Y_train = np.sum(P_train,axis=1)/M_train
Y_train = np.expand_dims(Y_train,axis=1)

with open(test_path, 'rb') as handle:
	testsets_dict = pickle.load(handle)

X_test = testsets_dict["params"]
P_test = testsets_dict["labels"]       
n_test_points, M_test = P_test.shape
Y_test = np.sum(P_test,axis=1)/M_test
Y_test = np.expand_dims(Y_test,axis=1)

smc_path = "TrainedModels/{}_{}_gpep.pickle".format(modelName,paramterName)

DO_TRAIN = True
if DO_TRAIN:
	smc = smMC_GPEP(modelName, paramterName)
	smc.load_train_data(X_train, Y_train, M_train)
	smc.load_test_data(X_test, Y_test)

	smc.fit()

	smc_dict = {"smc":smc}
	

	with open(smc_path, 'wb') as handle:
		pickle.dump(smc_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

	smc.predictive_results(len(params))
else:
	with open(smc_path, 'rb') as handle:
		smc_dict = pickle.load(handle)
	smc = smc_dict["smc"]
	smc.predictive_results(len(params))

