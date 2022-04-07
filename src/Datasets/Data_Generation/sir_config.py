def sir_config_details(list_params):
    D = {}

    D["model"] = "SIR.psc"
    D["variables"] =  ["S","I","R"]
    D["parameters"] = ["beta","gamma"]
    D["state_space_dim"] = 3
    D["populationsize"] =100
    D["param_space_dim"] = len(list_params)
    D["params"] = list_params
    D["time_step"] = 0.5
    D["n_steps"] = 240

    if list_params == ["beta","gamma"]:
        D["params_min"] = [0.005,0.005]
        D["params_max"] = [0.3,0.2]

        D["n_steps_param"] = 16
        D["n_trajs"] = 10
    elif list_params == ["beta"]:
        D["params_min"] = [0.005]
        D["params_max"] = [0.3]

        D["n_steps_param"] = 200
        D["n_trajs"] = 10
    elif list_params == ["gamma"]:
        D["params_min"] = [0.005]
        D["params_max"] = [0.2]

        D["n_steps_param"] = 200
        D["n_trajs"] = 10
    else:
        print("Config with {} as parameters not defined!!".format(list_params))
    D["n_combination"] = D["n_steps_param"]**D["param_space_dim"]
    
    return D