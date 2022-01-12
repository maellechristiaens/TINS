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



def synchrony_over_time_transitions(transitions, mean_sync, rem_to_sws=True):
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



def correlation(var1, var2, name_var1, name_var2, type_of_neurons):
    '''
    MC 12/01/22
    This function plots the linear regression between 2 pairs of variables
    Inputs: 
        var1/var2: the variables (in NeurotimeSeries)
        name_var1/name_var2: the name for the axis legend (in str)
        type_of_neurons: if it's pyramidal or interneurons 
    Outputs: 
        A graph of the linear regression 
    '''
    
    plt.figure()
    g = sns.regplot(var1.restrict(states['Rem']).values, 
                    var2.restrict(states['Rem']).values, 
                    scatter_kws={'alpha':0.1, 's':5}, line_kws={'lw':5},   x_ci='sd', color='orange')

    sns.regplot(var1.restrict(states['sws']).values, 
                var2.restrict(states['sws']).values, 
                scatter_kws={'alpha':0.1, 's':5}, line_kws={'lw':5}, x_ci='sd', color='grey')

    sns.regplot(var1[100:-100].restrict(states['wake']).values, 
                var2[100:-100].restrict(states['wake']).values, 
                scatter_kws={'alpha':0.1, 's':5}, line_kws={'lw':5}, x_ci='sd')

    g.set(xlabel = name_var1, ylabel = name_var2,
          title= f'Correlation between {name_var1} and {name_var2} in BLA {type_of_neurons}')
    g.legend(labels=['Rem', 'sws', 'Wake'])
    
    
def variation_across_session (states, variables, labels, xmin=None, xmax=None):
    '''
    MC 12/01/22
    This function plots a variable across the whole session
    Inputs:
        states: the states during the session (REM, SWS, WAKE)
        variables: the different variables to plot (in neurotimeseries)
        labels: the names of the different variables plotted
        xmin/xmax: the limits of the shown window if we want to see a particular spot. By default = None
    Outputs:
        A graph of the variable across time with the REM and NREM epochs shown
    
    '''
    
    plt.figure() #create a new figure 
    bk.plot.intervals(states['Rem']) #plot the REM intervals in orange
    bk.plot.intervals(states['sws'], col='grey') #plot the NREM intervals in grey
    
    for (var,label) in zip(variables, labels): #for each variable we want to plot 
        plt.plot(var[15000:].as_units('s'), label=label, color = 'g') #plot it with its corresponding label, in a random color
        
    if xmin: #if we want to look at a certain time of the session
        plt.xlim(xmin, xmax) #restrict the window of the figure to this certain time
        
    plt.legend()