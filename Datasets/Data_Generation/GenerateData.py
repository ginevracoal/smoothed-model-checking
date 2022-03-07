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

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--modelName", type=str,default='SIR', help="Model name(without suffix .psc")

    D = fill_dictionary_with_parameters(parser.parse_args())
    
    # Pass read params and intitialize generator
    G = Generator(D)

    # Generator
    G.generate()

    # Save data to folder
    G.save_dataset_values()

    
    

    
