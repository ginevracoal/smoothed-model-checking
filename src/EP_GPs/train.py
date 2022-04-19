import os, sys
import numpy as np
import pickle5 as pickle

sys.path.append(".")
from paths import *
from EP_GPs.smMC_GPEP import *

models_path = os.path.join("EP_GPs", models_path)
plots_path = os.path.join("EP_GPs", plots_path)

for filepath, train_filename, val_filename, params_list, math_params_list in case_studies:

    with open(os.path.join(data_path, filepath, train_filename+".pickle"), 'rb') as handle:
        train_data = pickle.load(handle)
    
    paramterName = params_list[0] if len(params_list)==1 else ''.join(params_list)

    x_train = train_data["params"]
    p_train = train_data["labels"]
    n_train_points, m_train = p_train.shape
    y_train = np.sum(p_train,axis=1)/m_train
    y_train = np.expand_dims(y_train,axis=1)

    with open(os.path.join(data_path, filepath, val_filename+".pickle"), 'rb') as handle:
        val_data = pickle.load(handle)
        
    x_test = val_data["params"]
    p_test = val_data["labels"]       
    n_test_points, m_test = p_test.shape
    y_test = np.sum(p_test,axis=1)/m_test
    y_test = np.expand_dims(y_test,axis=1)

    DO_TRAIN = True
    if DO_TRAIN:
        smc = smMC_GPEP(filepath, paramterName)
        smc.load_train_data(x_train, y_train, m_train)
        smc.load_test_data(x_test, y_test)

        smc.fit()
        smc.save(filepath=models_path, filename=train_filename)
        post_mean, q1, q2 = smc.make_predictions(val_data["params"])

        # post_mean, q1, q2, evaluation_dict = smc.eval_gp(val_data)

    else:
        smc.load(filepath=models_path, filename=train_filename)

        post_mean, q1, q2 = smc.make_predictions(val_data["params"])
        # post_mean, q1, q2, evaluation_dict = smc.eval_gp(val_data)

    if len(params_list)<=2:

        fig = plot_posterior(params_list=params_list, math_params_list=math_params_list, train_data=train_data,
            test_data=val_data, val_data=val_data, post_mean=post_mean, q1=q1, q2=q2)

        os.makedirs(os.path.dirname(plots_path), exist_ok=True)
        fig.savefig(plots_path+f"{out_filename}.png")
