def prgeex_config_details(list_params, train_flag):

	D = {}

	D["model"] = "PrGeEx.psc"
	D["variables"] =  ["PLac","RNAP","PLacRNAP","TrLacZ1","RbsLacZ","TrLacZ2","Ribosome","RbsRibosome","TrRbsLacZ","LacZ","dgrLacZ","dgrRbsLacZ"]
	D["parameters"] = ["k1","k2","k3","k4","k5","k6","k7","k8","k9","k10","k11"]

	D["state_space_dim"] = 12
	D["populationsize"] = 1

	D["param_space_dim"] = len(list_params)
	D["params"] = list_params
	D["time_step"] = 5
	D["n_steps"] = 420

	if list_params == ["k2","k7"]:
		D["params_min"] = [10,0.45]
		D["params_max"] = [10000,450]
		if train_flag:
			D["n_steps_param"] = 50
			D["n_trajs"] = 50
		else:
			D["n_steps_param"] = 20
			D["n_trajs"] = 1000
	elif list_params == ["k2"]:
		D["params_min"] = [10]
		D["params_max"] = [100000]
		if train_flag:
			D["n_steps_param"] = 500
			D["n_trajs"] = 50
		else:
			D["n_steps_param"] = 1000
			D["n_trajs"] = 1000
	elif list_params == ["k7"]:
		D["params_min"] = [0.45]
		D["params_max"] = [4500]
		if train_flag:
			D["n_steps_param"] = 500
			D["n_trajs"] = 50
		else:
			D["n_steps_param"] = 1000
			D["n_trajs"] = 1000
	else:
		print("Config with {} as parameters not defined!!".format(list_params))
	
	D["n_combination"] = D["n_steps_param"]**D["param_space_dim"]

	return D