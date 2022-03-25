# filepath, train_filename, val_filename, params_list

data_paths = [
    ["Poisson", "Poisson_DS_46samples_1obs_lambda", None, ["lambda"]], 
    ["Poisson", "Poisson_DS_46samples_5obs_lambda", None, ["lambda"]],
    ["Poisson", "Poisson_DS_46samples_10obs_lambda", None, ["lambda"]],
    ["SIR", "SIR_DS_200samples_10obs_beta", "SIR_DS_20samples_5000obs_beta", ["beta"]],
    ["SIR", "SIR_DS_200samples_10obs_gamma", "SIR_DS_20samples_5000obs_gamma", ["gamma"]],
    ["SIR", "SIR_DS_256samples_10obs_betagamma", "SIR_DS_256samples_5000obs_betagamma", ["beta", "gamma"]],
    ["PrGeEx", "PrGeEx_DS_25samples_10obs_k2", "PrGeEx_DS_25samples_100obs_k2", ["k2"]],
    ["PrGeEx", "PrGeEx_DS_25samples_10obs_k7", "PrGeEx_DS_25samples_100obs_k7", ["k7"]],
    ["PrGeEx", "PrGeEx_DS_100samples_10obs_k2k7", "PrGeEx_DS_100samples_100obs_k2k7", ["k2", "k7"]],
    ["PhosRelay", "PhosRelay_DS_200samples_10obs_k1", "PhosRelay_DS_12samples_1000obs_k1", ["k1"]],
    ["PhosRelay", "PhosRelay_DS_1000_latin_samples_10obs_k1k2k3", "PhosRelay_DS_216_latin_samples_500obs_k1k2k3",
        ["k1","k2","k3"]],
    ["PhosRelay", "PhosRelay_DS_100000_latin_samples_10obs_k0k1k2k3k4", 
        "PhosRelay_DS_1024_latin_samples_500obs_k0k1k2k3k4", ["k0","k1","k2","k3","k4"]],
]

models_path = 'models/'
plots_path='plots/'
data_path = '../data/'
