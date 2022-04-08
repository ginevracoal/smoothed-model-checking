def relay_config_details(list_params):

    D = {}

    D["model"] = "PhosRelay.psc"
    D["variables"] =  ["B","L1","L2","L3","L1P","L2P","L3P"]
    D["parameters"] = ["k0","k1","k2","k3","k4"]

    D["state_space_dim"] = 7
    D["populationsize"] = 1
    # Sampling values
    D["time_step"] = 1
    D["n_steps"] = 1200
    D["param_space_dim"] = len(list_params)
    D["params"] = list_params

    if len(list_params) == 1:

        D["params_min"] = [0.1]
        D["params_max"] = [2]
        D["n_steps_param"] = 30
        D["n_trajs"] = 2000

    elif len(list_params) == 3:

        D["params_min"] = [0.1,0.1,0.1]
        D["params_max"] = [2,2,2]
        D["n_steps_param"] = 5
        D["n_trajs"] = 2000

    elif len(list_params) == 5:

        D["params_min"] = [0.01, 0.1,0.1,0.1,0.5]
        D["params_max"] = [0.2,2,2,2,5]
        D["n_steps_param"] = 3
        D["n_trajs"] = 2000

    else:
        print("Config with {} as parameters not defined!!".format(list_params))

    D["n_combination"] = D["n_steps_param"]**D["param_space_dim"]
    
    return D