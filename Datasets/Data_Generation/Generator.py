# Generics
import configparser
import numpy as np
import itertools
import stochpy
import os
import pandas as pd
import pickle
import io
import pathlib
from tqdm import tqdm
from contextlib import redirect_stdout
from pathlib import Path
from datetime import datetime
import json


class Generator(object):
   
    def __init__(self, argsDictionary):
        """
        It reads the dictionary passed by GenerateData (argsDictionary)
        and also read the config file for the model(e.g. SIR.cfg) specified where the
        attributes of the simulation are written (e.g. beta and gamma with
        their relative min & max values).
        
        It also create the folder structure necessary for the experiment
        and setup the stochpy simulation
        """
    
        # Extract params from config file, set the model filepath
        self.D = self.castParams(self.getModelParameters(argsDictionary))

        # Data folders organization
        self.setup_folders()
      
        # Set stochpy attributes to perform the simulation
        self.stoch_mod = stochpy.SSA(IsInteractive=False)
        self.stoch_mod.Model(self.D['model'],  self.D['simFolder'])
        self.stoch_mod.output_dir =  self.D['ResultsFolder']
        self.stoch_mod.temp_dir = self.D['TempFolder']
        self.dataframes_dir = self.D['DataframeFolder']

        # Print the parameters for debugging purposes
        self.prettyPrintDictionaryParameters()    

    def setup_folders(self):
        
        now = self.D['modelName'] + datetime.now().strftime("_%H_%M_%S")
        self.D['experimentFolder'] = self.D['todayFolder']/now
        self.D['experimentFolder'].mkdir(parents=True, exist_ok=True)
        
        self.D['ResultsFolder'] = self.D['experimentFolder'] / "Results"
        self.D['TempFolder'] = self.D['experimentFolder'] / "Temp"
        self.D['DataframeFolder'] = self.D['experimentFolder'] / "Dataframes"
        
        
        self.D['ResultsFolder'].mkdir(parents=True, exist_ok=True)
        self.D['TempFolder'].mkdir(parents=True, exist_ok=True)
        self.D['DataframeFolder'].mkdir(parents=True, exist_ok=True)
        
    def prettyPrintDictionaryParameters(self):
         #Pretty print parameters
        print( "MODEL FILE ---> ", self.stoch_mod.model_file,  " \nMODEL DIR ---> ", self.stoch_mod.model_dir ," \nOUTPUT DIR ---> ", self.stoch_mod.output_dir , " \nTEMP DIR ---> ", self.stoch_mod.temp_dir )
        print("\n".join("{}\t{}".format(k, v) for k, v in self.D.items()))
        
    def getModelParameters(self, argsDictionary):
        """
        Extract parameters from configuration file.
        Merge the config file with the 
        """      
        path =  argsDictionary['configFolder'] / (argsDictionary['modelName'] + '.cfg')
        cp = configparser.ConfigParser()
        cp.read(path)
        D = {**dict(cp.items('model')), **argsDictionary}
        return D
    
    
    def is_number(self,a):
        return a.replace('.','',1).isdigit()
    
    def stringListToList(self,s):
        return s.strip('[').strip(']').split(',')
    
    def castParams(self, paramsDictionary):
        
        for key, value in paramsDictionary.items(): 
            #print(key, ' -> ', value, ' -> ', type(value))
            if isinstance(value, pathlib.PosixPath) is False:
                # if it is a list ...
                if("]" in value):
                    # Converting string to list
                    L = self.stringListToList(value)
                    paramsDictionary[key]  = [float(x) if self.is_number(x) else x for x in L]
                # if it is a single value ...
                else:
                    # if it is a number
                    if(self.is_number(value)):
                        paramsDictionary[key] = float(value)           
        # Casting
        paramsDictionary['n_steps'] = int(paramsDictionary['n_steps'])
        paramsDictionary['state_space_dim'] = int(paramsDictionary['state_space_dim'])
        paramsDictionary['param_space_dim'] = int(paramsDictionary['param_space_dim'])
        paramsDictionary['n_trajs'] = int(paramsDictionary['n_trajs'])
        
        # Create grid of parameters for simulation
        paramsDictionary['paramsGrid'] = self.generateGrid(paramsDictionary)
        
        # How many set of parameters, How many points per set of states. 
        paramsDictionary['n_combination'] =  paramsDictionary['paramsGrid'].shape[0]
        
        # Times
        paramsDictionary['T'] = int(paramsDictionary['n_steps']*paramsDictionary['time_step'])

        return paramsDictionary
     
    def generateGrid(self,paramsDictionary):
        L = []
        for p, lower,upper in zip(paramsDictionary['params'],paramsDictionary['params_min'],paramsDictionary['params_max']):
            bounds = np.linspace(lower, upper, num=int(paramsDictionary['n_steps_param']))
            L.append(bounds) 
        cartesian_product = np.array(list(itertools.product(*L)))
        return cartesian_product
    
    # questo mancava nel codice di Paolo
    def set_parameters(self, set_param):
        for i in range(self.D["param_space_dim"]):
            self.stoch_mod.ChangeParameter(self.D["params"][i], set_param[i])

        

    def time_resampling(self, data):
        time_index = 0
        time_array = np.linspace(0, self.D['T'], num=self.D['n_steps']+1)
        new_data = np.zeros((time_array.shape[0], data.shape[1]))
        new_data[:, 0] = time_array
        for j in range(len(time_array)):
            while time_index < data.shape[0] - 1 and data[time_index + 1][0] < time_array[j]:
                time_index = time_index + 1
            if time_index == data.shape[0] - 1:
                new_data[j, 1:] = data[time_index, 1:]
            else:
                new_data[j, 1:] = data[time_index, 1:]
        return new_data
    
    
        
    def generate(self):
               
        traces = np.zeros((self.D['n_combination'], self.D['n_trajs'], self.D['n_steps'] , self.D['state_space_dim']))
        meta =  np.zeros((self.D['n_combination'], self.D['n_trajs'], self.D['param_space_dim'] ))
 
        extern_counter=0
        
        # ForEach set of params
        for param_set in tqdm(self.D['paramsGrid']): 
            print(param_set)
            self.set_parameters(param_set)
            X  = np.zeros((self.D['n_trajs'], self.D['n_steps'], self.D['state_space_dim']))
            Yp = np.zeros((self.D['n_trajs'],self.D['param_space_dim']))
            
            count = 0 
            
            for n in range(self.D['n_trajs']):
        
                # SIMULATE                   
                with redirect_stdout(io.StringIO()):
                    self.stoch_mod.DoStochSim(method="Direct", trajectories=1, mode="time", end=self.D['T'], quiet=True)    
                    self.stoch_mod.Export2File(analysis='timeseries', datatype='species',IsAverage=False, quiet=True)
                
                # READ SIMULATED RESULTS
                datapoint = pd.read_csv(filepath_or_buffer= self.stoch_mod.output_dir / Path(self.D['model'] + "_species_timeseries1.txt"), delim_whitespace=True, header=1)
                datapoint = datapoint.drop(labels="Reaction", axis=1).drop(labels='Fired', axis=1).to_numpy()#.drop("N",axis = 1).to_numpy()

                # PERFORM TRACES STANDARDIZATION THROUGH RESAMPLING
                new_point = self.time_resampling(datapoint)
                new_point = new_point[1:, 1:]
                X[count,:,:] = new_point
                
                # KEEP TRACK OF STARTING STATE/PARAM SET FOR EACH SET OF VALUES
                Yp[count,:] = param_set

                # SIMULATION COUNTER
                count += 1
        
            traces[extern_counter,:,:,:] = X
            meta[extern_counter,:,:] = Yp
            
            
            # GLOBAL COUNTER
            extern_counter += 1
            
        self.results   = np.copy(traces)
        self.etiquette = np.copy(meta)
                    
            
    def save_dataset_values(self):
        dataset_dict = {"X": self.results, "Y": self.etiquette}
        with open(self.dataframes_dir/Path(self.D['modelName'] +'_NUMPY.pkl'), 'wb') as handle:
            pickle.dump(dataset_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
            
        D_str = [dict([key, str(value)] for key, value in self.D.items())]
        with open(str(self.dataframes_dir/Path(self.D['modelName'] +'_parameters.json')), 'w') as handle:
            json.dump(D_str, handle,   indent=4)

            
            

           
        
        
    
    
    
    
    
    
    
    