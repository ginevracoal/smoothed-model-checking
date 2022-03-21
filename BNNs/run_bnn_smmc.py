from data_utils import *
from bnn_smmc import *

model_name = "PhosRelay"
nb_param = 3

list_param_names = ["k1","k2","k3"]
param_name = ''.join(list_param_names)
print(param_name)
casestudy_name = model_name+param_name

df_file_train, df_file_val = get_data_path(model_name, param_name, nb_param)

TRAIN_FLAG = True

n_hidden = 10
bnn = BNN_smMC(model_name, list_param_names, df_file_train, df_file_val, nb_param, n_hidden)

n_epochs = 10000
lr = 0.01
identifier = 1
bnn.run(n_epochs, lr, identifier, train_flag = TRAIN_FLAG)

print("TrainFlag = ", TRAIN_FLAG)
print("Train set: ", df_file_train)
print("Validation set: ", df_file_val)
print("model_name = {}, param_name = {}, n_hidden = {}, n_epochs = {}, lr = {}, identifier = {}".format(model_name, param_name, n_hidden, n_epochs, lr, identifier))