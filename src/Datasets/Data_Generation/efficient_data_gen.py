from GenerateData import *
from BooleanLabeler import *
from model_utils import *
import time
from contextlib import redirect_stdout
model_name = "PRDeg"
list_params = ["kprod","k1","k2","k3","k4","kdeg"]
latin_flag = True
train_flag = False
formula, position_dict = get_model_details(model_name)


gen = get_generator(model_name, list_params, latin_flag, train_flag)
gen.D["position_dict"] = position_dict
gen.D["formula"] = formula 
gen.D["T"] = gen.D["time_step"]*gen.D["n_steps"]

labeler = BooleanLabeler(gen.D)


input_data =  np.empty((gen.D['n_combination'], gen.D['n_trajs'], gen.D['param_space_dim'] ))
bool_labels = np.empty((gen.D["n_combination"], gen.D["n_trajs"]))

extern_counter=0

if gen.D["latinFlag"]:
    gen.D['paramsLatinGrid'] = gen.generateLatinGrid()
    sampled_parameters = gen.D['paramsLatinGrid']
else:
    gen.D['paramsGrid'] = gen.generateGrid()
    sampled_parameters = gen.D['paramsGrid']

start_time = time.time()
for param_set in tqdm(sampled_parameters): 
    print(param_set)
    gen.set_parameters(param_set)

    count = 0 
    
    for j in range(gen.D['n_trajs']):

        # SIMULATE                   
        with redirect_stdout(io.StringIO()):
            gen.stoch_mod.DoStochSim(method="Direct", trajectories=1, mode="time", end=gen.D['T'], quiet=True)    
            gen.stoch_mod.Export2File(analysis='timeseries', datatype='species',IsAverage=False, quiet=True)
        
        # READ SIMULATED RESULTS
        datapoint = pd.read_csv(filepath_or_buffer= gen.stoch_mod.output_dir / Path(gen.D['model'] + "_species_timeseries1.txt"), delim_whitespace=True, header=1)
        #print(datapoint)
        datapoint_cut = datapoint.drop(labels="Reaction", axis=1).drop(labels='Fired', axis=1).to_numpy()
        # PERFORM TRACES STANDARDIZATION THROUGH RESAMPLING
        trajectory = gen.time_resampling(datapoint_cut)[1:, 1:]

        bool_labels[extern_counter,j] = labeler.analyze(trajectory)


    input_data[extern_counter] = param_set
finish_time = time.time()-start_time
print("Time to generate dataset of {}x{}={} points: {}".format(gen.D["n_combination"], gen.D["n_trajs"], gen.D["n_trajs"]*gen.D["n_combination"], finish_time))
labeler.D["dataset_dict"] = {"params": input_data, "labels": bool_labels}
labeler.exportLabeledData()