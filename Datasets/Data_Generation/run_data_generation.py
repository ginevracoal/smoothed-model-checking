from GenerateData import *
from BooleanLabeler import *
from model_utils import *

model_name = "PhosRelay"
latin_flag = "True"
formula, position_dict = get_model_details(model_name)


if model_name == "PrGeEx":
	gen = run_trajectories_generation(model_name)
	gen.D["mean_data_dict"] = gen.traj_dict
	gen.D["n_trajs"] = 10
	gen.generate()
	
else:
	gen = run_trajectories_generation(model_name, latin_flag)

gen.D["traj_dict"] = gen.traj_dict
gen.D["position_dict"] = position_dict
gen.D["formula"] = formula 

labeler = BooleanLabeler(gen.D)
labeler.get_bool_labels()
labeler.exportLabeledData()

