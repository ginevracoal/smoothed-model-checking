def prdeg_config_details(list_params, train_flag):

    D = {}

    D["model"] = "PRDeg.psc"
    D["variables"] =  ["B","L1","L2","L3","L1P","L2P","L3P"]
    D["parameters"] = ["kprod","k1","k2","k3","k4", "kdeg"]

    D["state_space_dim"] = 7
    D["populationsize"] = 1
    # Sampling values
    D["time_step"] = 1
    D["n_steps"] = 600
    D["param_space_dim"] = len(list_params)
    D["params"] = list_params

    if list_params == ["k1"]:

        D["params_min"] = [0.1]
        D["params_max"] = [2]
        if train_flag:
            D["n_steps_param"] = 500
            D["n_trajs"] = 50
        else:
            D["n_steps_param"] = 1000
            D["n_trajs"] = 1000
    if list_params == ["kprod", "kdeg"]:

        D["params_min"] = [0.01,0.005]
        D["params_max"] = [0.2,0.1]
        if train_flag:
            D["n_steps_param"] = 50
            D["n_trajs"] = 50
        else:
            D["n_steps_param"] = 20
            D["n_trajs"] = 1000

    elif list_params == ["k1","k2","k3"]:

        D["params_min"] = [0.1,0.1,0.1]
        D["params_max"] = [2,2,2]
        if train_flag:
            D["n_steps_param"] = 20
            D["n_trajs"] = 20
        else:
            D["n_steps_param"] = 10
            D["n_trajs"] = 1000
    elif list_params == ["k1","k2","k3","k4"]:

        D["params_min"] = [0.1,0.1,0.1,0.5]
        D["params_max"] = [2,2,2,5]
        if train_flag:
            D["n_steps_param"] = 10
            D["n_trajs"] = 20
        else:
            D["n_steps_param"] = 8
            D["n_trajs"] = 1000
    elif list_params == ["kprod","k1","k2","k3","k4","kdeg"]:

        D["params_min"] = [0.01, 0.1,0.1,0.1,0.5,0.005]
        D["params_max"] = [0.2,2,2,2,5,0.1]
        if train_flag:
            D["n_steps_param"] = 10
            D["n_trajs"] = 20
        else:
            D["n_steps_param"] = 4
            D["n_trajs"] = 1000

    else:
        print("Config with {} as parameters not defined!!".format(list_params))

    D["n_combination"] = D["n_steps_param"]**D["param_space_dim"]
    
    return D