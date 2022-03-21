
# caricare ssa sims, bnn, gp binomial

# load validation data

validation_data = [
    ("SIR", "SIR_DS_20samples_5000obs_beta"),
    ("SIR", "SIR_DS_20samples_5000obs_gamma"),
    # ("SIR", "SIR_DS_256samples_10obs_betagamma"),
]

for val_filepath, val_filename in validation_data:

    with open(f"../Data/WorkingDatasets/{val_filepath}/{val_filename}.pickle", 'rb') as handle:
        data = pickle.load(handle)
    x_val, y_val, n_params, n_trials_val = build_binomial_dataframe(data)

    with torch.no_grad():

        x_test = []
        for col_idx in range(n_params):
            single_param_values = x_val[:,col_idx]
            x_test.append(torch.linspace(single_param_values.min(), single_param_values.max(), args.n_test_points))
        x_test = torch.stack(x_test, dim=1)

# fare grafici