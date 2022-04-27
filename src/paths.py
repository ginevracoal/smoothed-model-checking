# filepath, train_filename, val_filename, params_list, math_params_list

case_studies = [
    ["SIR", "SIR_DS_500samples_50obs_beta", "SIR_DS_1000samples_1000obs_beta", ["beta"], [r"$\beta$"]],
    ["SIR", "SIR_DS_500samples_50obs_gamma", "SIR_DS_1000samples_1000obs_gamma", ["gamma"], [r"$\gamma$"]],
    ["SIR", "SIR_DS_2500samples_50obs_betagamma", "SIR_DS_400samples_1000obs_betagamma", ["beta","gamma"], [r"$\beta$", r"$\gamma$"]],
    ["PrGeEx", "PrGeEx_DS_500samples_50obs_k2", "PrGeEx_DS_400samples_1000obs_k2", ["k2"], [r"$k2$"]],
    ## ["PrGeEx", "PrGeEx_DS_500samples_50obs_k7", "PrGeEx_DS_400samples_1000obs_k7", ["k7"], [r"$k7$"]],
    ["PrGeEx", "PrGeEx_DS_2500samples_50obs_k2k7", "PrGeEx_DS_400samples_1000obs_k2k7", ["k2","k7"], [r"$k2$", r"$k7$"]],
    ["PRDeg", "PRDeg_DS_500samples_50obs_k1", "PRDeg_DS_1000samples_1000obs_k1", ["k1"], [r"$k1$"]],
    # ["PRDeg", "PRDeg_DS_8000_latin_samples_20obs_k1k2k3", "PRDeg_DS_1000_latin_samples_1000obs_k1k2k3", ["k1","k2","k3"], [r"$k1$",r"$k2$",r"$k3$"]],
    # ["PRDeg", "PRDeg_DS_10000_latin_samples_20obs_k1k2k3k4", "PRDeg_DS_4096_latin_samples_1000obs_k1k2k3k4", ["k1","k2","k3","k4"], [r"$k1$",r"$k2$",r"$k3$",r"$k4$"]],
    ## ["PRDeg", "PRDeg_DS_2500samples_50obs_kprodkdeg", "PRDeg_DS_400samples_1000obs_kprodkdeg", ["kp", "kd"], [r"$k_p$", r"$k_d$"]],
]

data_path = 'data/'
plots_path='out/plots/'
models_path = 'out/models/'
