#!/usr/bin/env python2.7
# Run_HyperNEATBot.py
# Author: Shawn Beaulieu
# June 3rd, 2017

import sys, os
import random
import glob
import numpy as np
import pandas as pd
from functools import partial
from GeneticAlgorithms import GA


def single_shuffle(a, df=True):
    p = np.random.permutation(len(a))
    if df:
        return(a.iloc[p,:])
    else:
        return(a[p,:])

def load_data(files):
    for f in files:
        try:                                                                                                                    
            new_genomes = pd.read_csv(f, header=None)
            genomes = genomes.append(new_genomes)                                           
        except:                                                                                   
            genomes = pd.read_csv(f, header=None) 
    return(genomes)

if __name__ == '__main__':

    directory = os.getcwd()
    Parameters = {

        'popsize': 50,
        'generations': 1000,
        'blueprint': [5,8],
        'environments': 2,
        'elitism': 1.0,
        'crossover': False,
        'devo': True,
        'dropout': True,
        'metric': 'collective',
        'seed': int(sys.argv[1]),
        'folder': 'Experiment_15_DropoutSwarm'
    }


    GA(Parameters)
