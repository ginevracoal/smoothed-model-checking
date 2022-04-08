from GenerateData import *
from BooleanLabeler import *
from model_utils import *
import time

model_name = "PhosRelay"#"PrGeEx"
list_params = ["k1"]#["k7"]
latin_flag = False
formula, position_dict = get_model_details(model_name)

traj_time = time.time()
if model_name == "PrGeEx":
	gen = run_trajectories_generation(model_name, list_params, latin_flag)
	gen.D["mean_data_dict"] = gen.traj_dict
	#gen.D["n_trajs"] = 10
	#gen.generate()

else:
	gen = run_trajectories_generation(model_name, list_params, latin_flag)

traj_time = time.time()-traj_time

gen.D["traj_dict"] = gen.traj_dict
gen.D["position_dict"] = position_dict
gen.D["formula"] = formula 


labeler = BooleanLabeler(gen.D)
labels_time = time.time()
labeler.get_bool_labels()
labels_time = time.time()-labels_time
labeler.exportLabeledData()

print("Time needed to generate {} trajectories: {}".format(gen.D["n_trajs"]*gen.D["n_combination"], traj_time))
print("Time needed to label {} trajectories: {}".format(gen.D["n_trajs"]*gen.D["n_combination"], labels_time))
