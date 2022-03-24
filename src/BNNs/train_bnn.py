import sys
sys.path.append('../')
from paths import *
from BNNs.bnn import *

for filepath, train_filename, val_filename, params_list in data_paths:

    df_file_train = os.path.join(os.path.join(data_path, filepath, train_filename+".pickle"))
    df_file_val = os.path.join(os.path.join(data_path, filepath, val_filename+".pickle")) if val_filename else None

    model_name = filepath
    nb_param = len(params_list)
    param_name = ''.join(params_list)

    TRAIN_FLAG = True

    n_hidden = 10
    bnn = BNN_smMC(model_name, params_list, df_file_train, df_file_val, nb_param, n_hidden)

    n_epochs = 10000
    lr = 0.01
    identifier = 1
    bnn.run(n_epochs, lr, identifier, train_flag = TRAIN_FLAG)

    print("TrainFlag = ", TRAIN_FLAG)
    print("Train set: ", df_file_train)
    print("Validation set: ", df_file_val)
    print("model_name = {}, param_name = {}, n_hidden = {}, n_epochs = {}, lr = {}, identifier = {}".format(model_name, param_name, n_hidden, n_epochs, lr, identifier))