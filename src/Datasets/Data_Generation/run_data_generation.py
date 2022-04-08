from GenerateData import *
from BooleanLabeler import *
from model_utils import *
import time

model_name = "PRDeg"
list_params = ["k1","k2","k3"]
latin_flag = True
train_flag = False
formula, position_dict = get_model_details(model_name)

traj_time = time.time()
if model_name == "PrGeEx":
	gen = run_trajectories_generation(model_name, list_params, latin_flag)
	gen.D["mean_data_dict"] = gen.traj_dict
	#gen.D["n_trajs"] = 10
	#gen.generate()

else:
	gen = run_trajectories_generation(model_name, list_params, latin_flag, train_flag)
traj_time = time.time()-traj_time

if False:
	X = gen.traj_dict["X"]
	Y = gen.traj_dict["Y"]
	n_config, n_trajs, n_steps, n_species = X.shape
	timeline = np.linspace(0,600, n_steps)
	for i in range(n_config):
		fig = plt.figure()
		for j in range(n_trajs):
			#plt.plot(timeline, X[i,j,:,0], 'k')
			plt.plot(timeline, X[i,j,:,2], 'b')
			plt.plot(timeline, X[i,j,:,4], 'g')
			plt.plot(timeline, X[i,j,:,6], 'r')
		plt.xlabel("time")
		plt.tight_layout()
		fig.savefig("../Data/PrDegPlots/PRDeg_Trajs_{}_params={}.png".format(i,list_params))
		plt.close()


gen.D["traj_dict"] = gen.traj_dict
gen.D["position_dict"] = position_dict
gen.D["formula"] = formula 

print("Generaing Boolean labels...")

labeler = BooleanLabeler(gen.D)
labeler.set_traj_dict()
labels_time = time.time()
labeler.get_bool_labels()
labels_time = time.time()-labels_time
labeler.exportLabeledData()

print("Time needed to generate {} trajectories: {}".format(gen.D["n_trajs"]*gen.D["n_combination"], traj_time))
print("Time needed to label {} trajectories: {}".format(gen.D["n_trajs"]*gen.D["n_combination"], labels_time))
