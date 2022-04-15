def sir_config_details(list_params, train_flag):
    D = {}

    D["model"] = "SIR.psc"
    D["variables"] =  ["S","I","R"]
    D["parameters"] = ["beta","gamma"]
    D["state_space_dim"] = 3
    D["populationsize"] = 100
    D["param_space_dim"] = len(list_params)
    D["params"] = list_params
    D["time_step"] = 0.5
    D["n_steps"] = 240

    if list_params == ["beta","gamma"]:
        D["params_min"] = [0.005,0.005]
        D["params_max"] = [0.3,0.2]

        if train_flag:
            D["n_steps_param"] = 50
            D["n_trajs"] = 50
        else:
            D["n_steps_param"] = 20
            D["n_trajs"] = 1000

    elif list_params == ["beta"]:
        D["params_min"] = [0.005]
        D["params_max"] = [0.3]
        if train_flag:
            D["n_steps_param"] = 500
            D["n_trajs"] = 50
        else:
            D["n_steps_param"] = 1000
            D["n_trajs"] = 1000

    elif list_params == ["gamma"]:
        D["params_min"] = [0.005]
        D["params_max"] = [0.2]

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