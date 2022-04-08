from  Generator import Generator
import argparse
import os
import io
from pathlib import Path
from datetime import datetime
from relay_config import *
from sir_config import *
from prgeex_config import *
from prdeg_config import *

def add_flds_to_dictionary(D):
    
    # Set the folder structure 
    D['baseFolder'] = Path.cwd().parent
    D['modelFolder']  = D['baseFolder'] / "Models"
    D['configFolder'] = D['modelFolder'] / "ConfigFiles"
    D['simFolder']  = D['modelFolder'] / "SimFiles"
    D['dataFolder']  = D['baseFolder'] / "Data"
    
    # Create a folder for today's experiments
    today = datetime.now().strftime("%d_%m_%Y")
    D['todayFolder'] = D['dataFolder']/today
    D['todayFolder'].mkdir(parents=True, exist_ok=True)
    
    return D

def get_generator(model_name, list_params, latin_flag = False, train_flag = True):

    if model_name == "PhosRelay":
        D = relay_config_details(list_params)
    elif model_name == "SIR":
        D = sir_config_details(list_params, train_flag)
    elif model_name == "PrGeEx":
        D = prgeex_config_details(list_params)
    elif model_name == "PRDeg":
        D = prdeg_config_details(list_params, train_flag)
    else:
        D = {}
        print("{} not defined!!".format(model_name))

    D["modelName"] = model_name
    D["latinFlag"] = latin_flag

    D = add_flds_to_dictionary(D)

    G = Generator(D)

    return G

    
def run_trajectories_generation(model_name, list_params, latin_flag = False, train_flag = True):

    if model_name == "PhosRelay":
        D = relay_config_details(list_params)
    elif model_name == "SIR":
        D = sir_config_details(list_params, train_flag)
    elif model_name == "PrGeEx":
        D = prgeex_config_details(list_params)
    elif model_name == "PRDeg":
        D = prdeg_config_details(list_params, train_flag)
    else:
        D = {}
        print("{} not defined!!".format(model_name))

    D["modelName"] = model_name
    D["latinFlag"] = latin_flag

    D = add_flds_to_dictionary(D)
    
    # Pass read params and intitialize generator
    G = Generator(D)

    # Generator
    G.generate()

    return G
    
    

    
