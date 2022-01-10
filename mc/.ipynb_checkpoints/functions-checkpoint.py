import os
import sys

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
from tqdm import tqdm
import neuroseries as nts
import scipy as sp
import scipy.stats
import seaborn as sns

import bk.load
import bk.compute
import bk.plot
import bk.signal

import matplotlib as mpl
mpl.rcParams["svg.fonttype"] = "none"



def iterate_over_sessions(function): 
    '''
    MC 09/01/22
    Inputs:
        function: the function you want to apply to all you neurons 
    Outputs: 
        output_dict: a dictionary with the results associated with each session
        outputs: a list of all the results from the function 
    '''
    os.chdir('E:/DATA/GG-Dataset/') #change the directory to the base directory of the data
    session_index = pd.read_csv("relative_session_indexing.csv") #read the csv file where the 
                                                        #relative session indexes are stored 
    
    to_ignore = ["Rat08/Rat08-20130720", 
             "Rat08/Rat08-20130722", 
             "Rat08/Rat08-20130718",
             "Rat09/Rat09-20140408",
             "Rat09/Rat09-20140409",
             "Rat10/Rat10-20140619",
             "Rat10/Rat10-20140620",
             "Rat10/Rat10-20140622",
             "Rat10/Rat10-20140708",
             "Rat11/Rat11-20150309",
             "Rat11/Rat11-20150318",
             "Rat11/Rat11-20150319", 
             "Rat11/Rat11-20150324",
             "Rat11/Rat11-20150327",
             "Rat11/Rat11-20150401"]
    #all the sessions to ignore because they are bugged 
    
    output_dict = {} #initialisation of the dictionary with the results associated with each session
    outputs = [] #initialisation of all the results from the function
    
    for path in tqdm(session_index["Path"]): #go through all sessions in the csv file
        
        if path in to_ignore: #if the session is to ignore
            print(f'\n {path} THIS SESSION IS BULLSHIT \n')
            continue #skip it 
        try: #try to apply the function in argument to this session
            output = function(path) #store the results
            outputs.append(output)
            output_dict.update({bk.load.session : output})
        except: #if it's not applicable, return an error message 
            print(f'\n {path} ERROR IN THIS SESSION, TO WORK IT THROUGH \n')
            continue
            
        bk.load.current_session_linux(base_folder= 'E:/DATA/GG-Dataset/',local_path= local_path)
        states = bk.load.states() 

        if states['Rem'].tot_length()/1_000_000 < 100 or states['sws'].tot_length()/1_000_000 < 100: 
            print( f'\n {path} NOT ENOUGH REM \n' )
            break

        neurons,metadata = bk.load.spikes() 

        neurons_pyr_bla = neurons[(metadata['Type'] == 'Pyr') & (metadata['Region'] == 'BLA')]
        if len(neurons_pyr_bla) < 20: 
            print( f'\n {path} NOT ENOUGH NEURONS \n' )
            continue
        

    return output_dict, outputs