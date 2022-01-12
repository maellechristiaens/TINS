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




def mean_synchrony(neurons, asymetry=False, neurons2=False, window=1, smallbins=0.1, shift=1):
    '''
    MC 29/10/21
    This function computes the mean synchrony for neurons across time
    Inputs: 
        neurons: the neurons you want to compute the synchrony on
        smallbins: the size of the time bins used to compute the number 
                   of spikes per neuron in that bin (in sec), default = 0.1
        window: the size of the time window in which to compute the mean
                of synchrony among the neurons (in sec), default = 1
        shift: the shift between one window and the following (in sec), default = 1
        asymetry: if you want to compute this synchrony between 2 different populations of neurons, default = False
        neurons2: the second population of neurons, default = False
    Outputs:
        a neuroseries data frame with the mean of synchrony across time
    '''
    if asymetry: #if we want to compute the synchrony between 2 different populations of neurons
        neurons = np.concatenate((neurons, neurons2)) # concatenate the 2 populations of neurons
        l_neurons2 = len(neurons2) #take the length of the second population, it will be used to extract
                                   #only the part of the corrcoef matrix between the 2 populations and not between
                                   #themselves
    t,binned_neurons = bk.compute.binSpikes(neurons, binSize=smallbins) #count the number of spikes for each neuron in smallbins
    b = int(window/smallbins) #the number of small bins in one window
    c = int(shift/smallbins) #the number of small bins you have to shift each time
    mean_sync = np.zeros(int(binned_neurons.shape[1]/c)) #initialisation of the mean synchrony across time 
    start = int(0 + b/2) #where to start the computation of mean synchrony (you have to start at b because you
                       #compute corrcoef in [-b/2,b/2] windows)
    stop = int(binned_neurons.shape[1]-b/2) #where to end the computation of mean synchrony (at the end of the recording)
    for i,j in enumerate(tqdm(range(start, stop, c))): #for each window
        corrcoef = np.corrcoef(binned_neurons[:,int(j-b/2):int(j + b/2)]) #compute the corrcoef in a [-b/2:b/2] window
        if asymetry: 
            corrcoef = corrcoef[l_neurons2:,:l_neurons2] #extract only the corrcoef matrix between the 2 populations
        mean_sync[i] = np.nanmean(corrcoef) #compute the mean of corrcoef for each window, disregarding the nan
    t=t[::c] #extract t timings at c steps
    if len(t) == len(mean_sync):
        toreturn = nts.Tsd(t, mean_sync, time_units='s')
    else: 
        toreturn = nts.Tsd(t[:-1], mean_sync, time_units='s') #create a neuroseries frame with the means of synchrony
    return toreturn




def mean_synchrony_nan(neurons, asymetry=False, neurons2=False, window=2, smallbins=0.1, shift=1):
    '''
    MC 02/11/21
    This function computes the mean synchrony for neurons across time, but with NaN values changed to 0
    Inputs: 
        neurons: the neurons you want to compute the synchrony on
        smallbins: the size of the time bins used to compute the number 
                   of spikes per neuron in that bin (in sec), default = 0.1
        window: the size of the time window in which to compute the mean 
                of synchrony among the neurons (in sec), default = 2
        shift: the shift between one window and the following (in sec), default = 1
        asymetry: if you want to compute this synchrony between 2 different populations of neurons, default = False
        neurons2: the second population of neurons, default = False
    Outputs:
        a neuroseries data frame with the mean of synchrony across time
    '''
    if asymetry: #if we want to compute the synchrony between 2 different populations of neurons
        neurons = np.concatenate((neurons, neurons2)) # concatenate the 2 populations of neurons
        l_neurons2 = len(neurons2) #take the length of the second population, it will be used to extract
                                   #only the part of the corrcoef matrix between the 2 populations and not between
                                   #themselves
    t,binned_neurons = bk.compute.binSpikes(neurons, binSize=smallbins) #count the number of spikes for each neuron in smallbins
    b = int(window/smallbins) #the number of small bins in one window
    c = int(shift/smallbins) #the number of small bins you have to shift each time
    mean_sync = np.zeros(int(binned_neurons.shape[1]/c)) #initialisation of the mean synchrony across time 
    start = int(0 + b/2) #where to start the computation of mean synchrony (you have to start at b because you
                       #compute corrcoef in [-b/2,b/2] windows)
    stop = int(binned_neurons.shape[1]-b/2) #where to end the computation of mean synchrony (at the end of the recording)
    for i,j in enumerate(tqdm(range(start, stop, c))): #for each window
        corrcoef = np.corrcoef(binned_neurons[:,int(j-b/2):int(j + b/2)]) #compute the corrcoef in a [-b/2:b/2] window
        if asymetry: 
            corrcoef = corrcoef[l_neurons2:,:l_neurons2] #extract only the corrcoef matrix between the 2 populations
        corrcoef[np.isnan(corrcoef)] = 0 #all nan values are changed to 0
        mean_sync[i] = np.nanmean(corrcoef) #compute the mean of corrcoef for each window, disregarding the nan
    t=t[::c] #extract t timings at c steps
    toreturn = nts.Tsd(t[:-1], mean_sync, time_units='s') #create a neuroseries frame with the means of synchrony
    return toreturn




def mean_firing_rate(neurons, window=2, smallbins=0.1, shift=1):
    '''
    MC 02/11/21
    This function computes the mean firing rate of a neuron population across time
    Inputs: 
        neurons: the neurons you want to compute the mean firing rate on
        smallbins: the size of the time bins used to compute the number 
                   of spikes per neuron in that bin (in sec), default = 0.1
        window: the size of the time window in which to compute the mean 
                of firing rate among the neurons (in sec), default = 2
        shift: the shift between one window and the following (in sec), default = 1
    Outputs:
        a neuroseries data frame with the mean of firing rate across time
    '''
    t,binned_neurons = bk.compute.binSpikes(neurons, binSize=smallbins) #count the number of spikes for each neuron in smallbins
    b = int(window/smallbins) #the number of small bins in one window
    c = int(shift/smallbins) #the number of small bins you have to shift each time
    mean_fr = np.zeros(int(binned_neurons.shape[1]/c)) #initialisation of the mean firing rate across time 
    start = int(0 + b/2) #where to start the computation of mean synchrony (you have to start at b because you
                       #compute corrcoef in [-b/2,b/2] windows)
    stop = int(binned_neurons.shape[1]-b/2) #where to end the computation of mean synchrony (at the end of the recording)
    for i,j in enumerate(tqdm(range(start, stop, c))): #for each window
        mean_fr[i] = np.nanmean(binned_neurons[:,int(j-b/2):int(j + b/2)])/window #compute the mean of firing rate for each 
                                                                                  #window disregarding the nan
    t=t[::c] #extract t timings at c steps
    if len(t) == len(mean_fr):
        toreturn = nts.Tsd(t, mean_fr, time_units='s')
    else: 
        toreturn = nts.Tsd(t[:-1], mean_fr, time_units='s') #create a neuroseries frame with the means of firing rate
    return toreturn



def pourcent_active_neurons(neurons, window=1):
    '''
    MC 03/11/21
    This function computes the pourcentage of active neurons across time
    Inputs: 
        neurons: the neurons you want to compute the pourcentages on
        window: the size of the time window in which to compute the pourcentage of active neurons (in sec), default = 1
    Outputs:
        a neuroseries data frame with the pourcentage of active neurons across time
    '''
    t,binned_neurons = bk.compute.binSpikes(neurons, binSize=window) #count the number of spikes for each neuron in windows
    pourcent_active = np.zeros(int(binned_neurons.shape[1])) #initialisation of the pourcentages across time 
    for i,j in enumerate(tqdm(range(0, int(binned_neurons.shape[1]), window))): #for each window
        pourcent_active[i] = np.count_nonzero(binned_neurons[:,j])/len(neurons) #compute the pourcentage for each window
    t=t[::window] #extract t timings at shift steps
    toreturn = nts.Tsd(t, pourcent_active, time_units='s') #create a neuroseries frame with the pourcentages
    return toreturn



def normalization(neurons):
    '''
    MC 08/11/21
    this function normalizes the pourcentage of active cells to the mean firing rate
    Input : the neurons you want to compute this normalization on
    Output : a neuroseries timeframe of this normalized pourcentages of active cells 
    '''
    t,_ = bk.compute.binSpikes(neurons, binSize=1)
    pac = pourcent_active_neurons(neurons, window = 1) #compute the pourcent of active cells across time
    pac = bk.compute.nts_smooth(pac, 100, 50) #smoothing the results 
    mfr = mean_firing_rate(neurons, window=2, shift=1) #compute the mean FR across time 
    mfr = bk.compute.nts_smooth(mfr, 100, 50) #smoothing the results
    toreturn = [(i/j) for i,j in zip(pac.values, mfr.values)] #normalize the pac to the mfr
    toreturn = nts.Tsd(t, toreturn, time_units = 's') #create the tsd
    return toreturn
    
    
    

def synchrony_around_transitions(transitions, mean_sync):
    '''
    MC 13/12/21
    This function computes the mean of synchrony around transitions REM/sws or sws/REM, 
    and the mean of synchrony of the whole epochs considered (REM or sws)
    Inputs: 
        transitions: the timing of the transitions
        mean_sync: the mean synchrony across the session
    Output: a dataframe with the different means for each type 
    '''
    
    trans = {'Rem before transition': [], 
            'Rem after transition': [],
             'Rem': [],
            'sws before transition' : [], 
            'sws after transition': [],
            'sws': []}
    
    transitions_rem_sws = transitions[1][('Rem', 'sws')].index.values
    transitions_sws_rem = transitions[1][('sws', 'Rem')].index.values
    
    for i  in transitions_rem_sws:
        interval_rem = nts.IntervalSet(i-10_000_000, i, time_units='us')
        trans['Rem before transition'].append(np.mean(mean_sync.restrict(interval_rem).values))
        interval_sws = nts.IntervalSet(i, i+10_000_000, time_units='us')
        trans['sws after transition'].append(np.mean(mean_sync.restrict(interval_sws).values))

    for i  in transitions_sws_rem:
        interval_sws = nts.IntervalSet(i-10_000_000, i, time_units='us')
        trans['sws before transition'].append(np.mean(mean_sync.restrict(interval_sws).values))
        interval_rem = nts.IntervalSet(i, i+10_000_000, time_units='us')
        trans['Rem after transition'].append(np.mean(mean_sync.restrict(interval_rem).values))

    for i, j  in zip(states['Rem'].as_units('s').start, states['Rem'].as_units('s').end):
        interval = nts.IntervalSet(i,j, time_units='s')
        trans['Rem'].append(np.mean(mean_sync.restrict(interval).values))

    for i, j  in zip(states['sws'].as_units('s').start, states['sws'].as_units('s').end):
        interval = nts.IntervalSet(i,j, time_units='s')
        trans['sws'].append(np.mean(mean_sync.restrict(interval).values))

    trans = pd.DataFrame(dict([ (k,pd.Series(v)) for k,v in trans.items() ]))
    return trans



def plot_synchrony_over_time_transitions(transitions, mean_sync, rem_to_sws=True):
    '''
    MC 15/12/21
    This function draws the synchrony over time around the transitions
    TO CONTINUE
    Inputs:
        transitions: the timing of the transitions 
        mean_sync: the mean of synchrony over time 
        rem_to_sws: if we look at rem to sws transitions or sws to rem transitions (by default: rem_to_sws)
    Output:
    A figure with the synchrony over time drawn
    '''
    means =[]
    fig, ax = plt.subplots()
    for i  in transitions_rem_sws:
        interval = nts.IntervalSet(i-20_000_000, i+ 40_000_000, time_units='us')
        m = mean_sync_pyr_bla.restrict(interval).values
        means.append(m)
        plt.plot(m, color = 'grey', alpha = 0.3)
    a = mpatches.Rectangle((0,0), 20, 0.3, facecolor = 'orange', alpha = 0.5)
    ax.add_patch(a)
    mean = np.mean(means, axis=0)
    plt.plot(mean, color = 'r', linewidth=4)
    
    
def corr_coeff_without_nan(var1, var2):
    '''
    MC 25/12/21
    This function returns 
    Inputs:
        var1: the first variable we want to extract the non-NaN from
        var2: the second variable we want to extract the non-NaN from
    Outputs:
        var1: the first variable masked
        var2: the second variable masked
        mask: the mask (the position where there are non-NaN in the variables)
    
    '''
    
    mask = ~np.logical_or(np.isnan(var1), np.isnan(var2)) 
    #create a mask (a boolean array) that tells if there are NaN in 
    #either variable for each position

    var1 = np.compress(mask, var1)
    var2 = np.compress(mask, var2)
    #extract only the positions where there are non NaN in either variable
    
    return var1, var2, mask



def get_first_and_last_intervals(states):
    '''
    MC 29/12/2021
    This function returns the first and the last epochs of a given state
    
    Inputs: 
        states: the given state you want to first and last epochs from
    Outputs: 
        start: the first epoch of the given state
        last: the last epoch of the given state
    '''
    first = pd.DataFrame(dict([ (k,pd.Series(v)) for k,v in states.iloc[0].items() ])) 
    #take the first line of the given state, and return a dataframe of it 
    first = nts.IntervalSet(first) 
    #turn it into a neurotimeseries interval (to be compatible with .restrict)
    last = pd.DataFrame(dict([ (k,pd.Series(v)) for k,v in states.iloc[-1].items() ]))
    last = nts.IntervalSet(last)
    return first, last



def get_extended_sleep(states,sleep_th,wake_th):
    '''
    BK 03/01/22
    Return extended sleep session given state, sleep
    Inputs:
        states: the given states you want the extended sleep from
        sleep_th: the threshold of sleep
        wake_th: the threshold of wake
    Outputs:
        ext_sleep: the epochs of extended sleep 
    '''
    
    ext_sleep = states['sws'].union(states['Rem']).merge_close_intervals(wake_th,time_units = 's')
    ext_sleep = ext_sleep[ext_sleep.duration(time_units='s')>sleep_th].reset_index(drop = True)
    return ext_sleep




