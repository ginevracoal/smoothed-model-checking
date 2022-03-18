from  Generator import Generator
import argparse
import os
import io
from pathlib import Path
from datetime import datetime


def interpretParams(arguments):
    """
    Create dictionary from arguments
    """
    
    D = dict()
    for arg in vars(arguments):
        D[arg] = getattr(arguments,arg)
    return D

def fill_dictionary_with_parameters(arguments):
    
    # Read params passed by the script
    D = interpretParams(arguments)
    
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

def run_trajectories_generation(model_name, latin_flag = False):

    parser = argparse.ArgumentParser()
    parser.add_argument("--modelName", type=str,default=model_name, help="Model name(without suffix .psc")
    parser.add_argument("--latinFlag", type=str,default=latin_flag, help="Activates latin hyper-cube sampling")

    D = fill_dictionary_with_parameters(parser.parse_args())
    
    # Pass read params and intitialize generator
    G = Generator(D)

    # Generator
    G.generate()

    return G
    
    

    
