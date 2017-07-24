#!/usr/bin/env python
# HyperNeatBot.py
# Author: Shawn Beaulieu
# June 6th, 2017

import random
import copy
import pickle
import ast
import numpy as np
import constants as c
import matplotlib.pyplot as plt
from pyrosim import PYROSIM
from robot import ROBOT
from individual import INDIVIDUAL
from environments import ENVIRONMENTS
from peas.networks import rnn
from peas.methods import neat

#First population of parents

class HyperNEATBot(object):

    #def evaluate(self, network, task_env, verbose=True):
    def evaluate(self, network, verbose=True):
        # Grab the network architecture from converted HyperNEAT genome.
        # Pass said genome into INDIVIDUAL() to create a new robot instance.

        # Evolving Masks with CPPNs:
        #fitness = {}
        #for task in range(2):
            #epigenome = network.cm
            #epigenome = epigenome.clip(min=0.0)
        
        # Dual Tasks: 
	#    genome = network.cm
        #    milieu = ENVIRONMENTS()
        #    agent = INDIVIDUAL(genome, targetGenome, devo=True)
        #    agent.Start_Evaluation(milieu.envs[task], pp=False, pb=True, env_tracker=task)
        #    agent.Compute_Fitness()
        #    fitness['{0}'.format(task)] = agent.Print_Fitness()
        
        #totalFitness = np.sum(fitness.values())/2

        # Vanilla Implementaton:
        genome = network.cm
        milieu = ENVIRONMENTS()
        agent = INDIVIDUAL(genome, devo=False)
        agent.Start_Evaluation(milieu.envs[0], pp=False, pb=True, env_tracker=0)
	agent.Compute_Fitness()
	fitness = agent.Print_Fitness()

        # Save the fittest individuals for later use:

        if fitness >= 0.20:
            with open("task1_HN_genomes.txt", "a+") as genomefile:
                genomefile.write(str(genome.tolist()))
                genomefile.write("\n")
            with open("task1_HN_fitness.txt", "a+") as fitfile:
                fitfile.write(str(fitness))
                fitfile.write("\n")
            #for task in range(1,3):
            #    with open("MAML_T{0}_fitness.txt".format(task), "a+") as fitFile:
            #        fitFile.write(str(fitness[str(task-1)]))
            #        fitFile.write("\n")

        # Print the fitness of each robot to the terminal

        if verbose:
            print("Fitness = %s" % (fitness))
        return {'fitness': fitness}
    
    def solve(self, network):
        return(self.evaluate(network)['fitness'] > 1.0)
