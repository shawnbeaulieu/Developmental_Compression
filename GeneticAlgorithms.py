#!/usr/bin/env python2.7
# GeneticAlgorithms.py
# Author: Shawn Beaulieu
# February, 2018

"""
Under Construction:

A suite of genetic algorithms controlled by the Run_GA.py file. GAs include:
simple GA, OpenAIES, .... Made for testing developmental compression

"""

from __future__ import division

import os
import copy
import math
import random
import pickle
import numpy as np
import pandas as pd
import constants as c
from robot import ROBOT
from pyrosim import PYROSIM
from functools import partial
from individual import INDIVIDUAL
from environments import ENVIRONMENTS
from pathos.multiprocessing import ProcessingPool as Pool

#First population of parents

def Preserve(data, filename):
    #df_data.to_csv(filename, mode='a', sep=",", header=None, index=None)
    directory = os.getcwd()
    filename = "{0}/{1}".format(directory, filename)
    df_data = pd.DataFrame(data)

    try:
        df_data.to_csv(filename, mode='a', sep=",", header=None, index=None)
    except:
        # If no such file exists (new experiment) create it:
        df_data.to_csv(filename, sep=",", header=None, index=None)

def Generate(mean, std, dimensions, tensor):
    if tensor == 'genome':
        l = -1.0
        u = 1.0
    elif tensor == 'dropout':
        l = 0.0
        u = 1.0
    return(np.clip(np.random.normal(mean, std, dimensions), l, u))

def Mean_Gradient(z, u, s):
    return((z-u)/5)

def Sigma_Gradient(z, u, s):
    return(((z-u)**2 - s**2)/s**3)


def Compute_Gradients(matrices, scores):
    """
    Population based metric for approximating the gradient with respect to
    network generating parameters (NES)

    """

    # First compute mean and standard deviation across network features:
    means = np.mean(np.array(matrices), axis=0)
    stds = np.std(np.array(matrices), axis=0)
    # Initialize approximate gradients
    means_gradient = np.zeros_like(matrices[0])
    stds_gradient = np.zeros_like(matrices[0])

    for child in zip(matrices, scores):
        means_gradient += Mean_Gradient(child[0], means, stds)*child[1]
        stds_gradient += Sigma_Gradient(child[0], means, stds)*child[1]

    means_gradient /= len(matrices)
    stds_gradient /= len(matrices)
    return(means, stds, means_gradient, stds_gradient)

def Fill_Matrix(size, mean, std, mean_grad, std_grad, a):
    dummy = np.zeros(size)
    for i in range(size[0]):
        for j in range(size[1]):
            # Fixed variance:
            dummy[i,j] = np.random.normal(mean[i,j] + a*mean_grad[i,j], 5)
            #dummy[i,j] = \
            #    np.random.normal(mean[i,j]+a*mean_grad[i,j], std[i,j]+a*std_grad[i,j])

    return(dummy)

def Rank(matrices, scores):

    low = -int(len(scores)/2)
    high = int(len(scores)/2)
    ranked = np.argsort(scores)

    adj_scores = list(range(low+1, high))
    adj_scores = [s/100 for s in adj_scores]
    sorted_matrices = [matrices[r] for r in ranked]
    
    return(sorted_matrices, adj_scores)


def Initialize_Tensor(population, num_envs, layer_names, dropout=False):
    return_list = []
    for p in range(population):
        new_entry = {str(e):{'genome':{}, 'dropout':{}} for e in range(num_envs)}
        for e in range(num_envs):
            if dropout:
                for tensor in new_entry[str(e)].keys():
                    new_entry[str(e)][tensor] =  \
                        {"{0}to{1}".format(i,o): Generate(0,1,(i,o), tensor) for i,o in layer_names}
            # Remove dropout for oth environment (no development)
                new_entry['0']['dropout'] = {}
            else:
                new_entry[str(e)]['genome'] =  \
                    {"{0}to{1}".format(i,o): Generate(0,1,(i,o), tensor='genome') for i,o in layer_names}
        
        return_list.append(new_entry)

    return(return_list)
    
def Inspect_Swarm(E, children, scores, layer2layer, tensor):
    mean_grads = dict()
    std_grads = dict()
    means = dict()
    stds = dict()

    for e in E:

        mean_grads[str(e)] = {}
        std_grads[str(e)] = {}
        means[str(e)] = {}
        stds[str(e)] = {}
        for IO in layer2layer:
            layer = "{0}to{1}".format(IO[0], IO[1])
            matrices = [c[str(e)][tensor][layer] for c in children]
            # Compute mean, std, and corresponding gradients:
            a,b,c,d = Compute_Gradients(matrices, scores)

            (means[str(e)][layer], stds[str(e)][layer],
                mean_grads[str(e)][layer], std_grads[str(e)][layer]) = (a,b,c,d)

    return(means, stds, mean_grads, std_grads)


class GA():

    """
    Simple genetic algorithm

    """
    P = {

        'popsize': 200,
        'generations': 1000,
        'blueprint': [5,8],
        'environments': 4,
        'elitism': 0.25,
        'crossover': False,
        'devo': False,
        'dropout': False,
        'metric': 'atomic',
        'seed': 0,
        'folder': 'Data'
    }    

    def __init__(self, parameters={}):

        # Use dictionary of parameters to class variables of the same name:
        self.__dict__.update(GA.P, **parameters)
        multiplier = max(1, int(round(self.popsize*self.elitism)))
        
        self.parents = [0]*multiplier
        self.parent_scores = [0]*multiplier
        self.champion = 0
        self.directory = os.getcwd()

        new_path = "{0}/{1}".format(self.directory, self.folder)
        try:
            os.makedirs(new_path)
        except OSError:
            if not os.path.isdir(new_path):
                raise

        # Set up multiprocessing:     
        #cpus = multiprocessing.cpu_count()
        self.pool = Pool()

        # Allow for arbitrarily many layers. Ordered dimensions saved to list. Can easily index dictionary (children):
        self.layer2layer = zip(self.blueprint[0:len(self.blueprint)-1], self.blueprint[1:len(self.blueprint)])
            
        # Genomes now composed of dictionaries, where each entry specifies connections between layers
        if self.devo:
            self.children = Initialize_Tensor(self.popsize, self.environments+1, self.layer2layer, self.devo)

        else:
            self.children = Initialize_Tensor(self.popsize, 1, self.layer2layer)
        # Genomes are now lists of dictionaries. Each dictionary entry corresponds to a sheet in the genetic tensor
        # For every sheet there is a network defined by self.blueprint
        # Mutation first randomly selects a sheet, then randomly selects a layer, then randomly selects a synapse      

        self.Evolve()

    def Evolve(self):

        self.g = 0

        for g in range(self.generations):
            print("Generation {0}: High Score = {1}".format(g, self.champion))
            # Evaluate in parallel
            self.child_scores = self.pool.map(self.Evaluate, self.children)
            # Find new champion
            if self.child_scores[np.argmax(self.child_scores)] > self.champion:
                self.champion = self.child_scores[np.argmax(self.child_scores)]
            
            # Traditional individul metric? Or population based NES metric?
            if self.metric == 'atomic':
                self.Selection()
                self.Spawn()

            elif self.metric == 'collective':

                # Rank scores (batch normalization):
                self.children, self.child_scores = Rank(self.children, self.child_scores)

                if self.devo:
                    E = self.environments+1
                else:
                    E = 1

                if self.dropout: 
                    self.means, self.stds, self.mean_grads, self.std_grads = \
                        Inspect_Swarm(range(E), self.children, self.child_scores, self.layer2layer, 'genome')
                    self.d_means, self.d_stds, self.d_mean_grads, self.d_std_grads = \
                        Inspect_Swarm(range(1,E), self.children, self.child_scores, self.layer2layer, 'dropout')
  
                else:
                    self.means, self.stds, self.mean_grads, self.std_grads = \
                        Inspect_Swarm(range(E), self.children, self.children_scores, self.layer2layer, 'genome')
                # Replenish swarm  
                self.New_Swarm(E)

                self.g += 1
               
   
        print("End of evolution: High Score = {0}".format(self.champion))

    def Evaluate(self, genome):
        """
        (Parallelized) evaluation in Pyrosim. Customize tasks in environment.py and
        environments.py.   
    
        """
        # Genome is a dictionary of dictionaries. Base is the first such dictionary.
        #base = genome["0"]
        # Establish environments
        milieu = ENVIRONMENTS()
        # Create individual
        fitness = []
        for e in range(self.environments):
            schedule = [True, False]
            if self.devo:
                target = genome[str(e+1)]['genome']
                dropout = genome[str(e+1)]['dropout']
                base = genome["0"]['genome']
                schedule = [True, False]
            else:
                base = genome["0"]
                target = 0.0
                dropout = 1.0
                schedule = [False]
            for s in schedule:
                agent = INDIVIDUAL(genome=base,target_genome=target, dropout=dropout, 
                                blueprint=self.layer2layer,devo=s, gens=self.generations, g=self.g)

                agent.Start_Evaluation(milieu.envs[e], pp=False, pb=True, env_tracker=e)
                agent.Compute_Fitness()
                fitness += [agent.Print_Fitness()]

        with open("{0}/{1}/Fitness_History_Seed{2}.csv".format(self.directory, self.folder, self.seed), "a+") as fitfile:
            fitfile.write(",".join([str(f) for f in fitness]))
            fitfile.write("\n")
       
        total_fitness = sum(fitness)/len(fitness)

        if fitness[1] > 0.20 and fitness[-1] > 0.20:
            with open("{0}/{1}/Matrices_Seed{1}.csv".format(self.directory, self.folder, self.seed), "a+") as fitfile:
                fitfile.write(str(genome))
                fitfile.write("\n")
        return(total_fitness)

    def Selection(self):
         """
         Locate the worst performing individuals of the previous generation
         and replace them with the best performing individuals of the current
         generation
      
         """
         best = max(1, int(round(self.popsize*self.elitism)))
         # Keep only x percent of the "best" individuals
         best_indices = np.argsort(self.child_scores)[-best:]
         #Preserve(self.children[indices[-1].reshape(1,-1), "Champion_Weights_{0}.csv".format(self.seed))  
         # Populate the "parents" with the fittest individuals: 

         for i in best_indices:
             worst = np.argmin(self.parent_scores)
             if self.child_scores[i] > self.parent_scores[worst]:
                 # Replace parent (score and genome) if better:
                 self.parent_scores[worst] = self.child_scores[i]
                 self.parents[worst] = self.children[i]
           
    def Spawn(self):
        """
        After selecting for high perfoming individuals, generate a new population whose 
        parameters depend on the values of those of the prior generation.

        """        

        self.children = []

        if self.crossover:
            while len(self.children) < self.popsize:
                # Randomly choose two parents (xx and xy)
                
                xx_idx = random.choice(range(len(self.parents)))
                xy_idx = random.choice(range(len(self.parents)))

                while xy_idx == xx_idx:
                    xy_idx = random.choice(range(len(self.parents)))
                              
                xx = self.parents[xx_idx]
                xy = self.parents[xy_idx]
                child = dict()
                for e in range(2):
                    child[str(e)] = {"{0}to{1}".format(i,o): np.zeros((i,o)) for i,o in self.layer2layer}
                    for IO in self.layer2layer:
                        layer = "{0}to{1}".format(IO[0], IO[1])
                        # Randomly choose which genes to keep in offspring:
                        xx_mask = np.random.choice(range(2), IO)
                        xy_mask = abs(xx_mask-1)
                        child[str(e)][layer] += xx[str(e)][layer]*xx_mask
                        child[str(e)][layer] += xy[str(e)][layer]*xy_mask
                         
                        i = random.choice(range(IO[0]))
                        o = random.choice(range(IO[1]))
                        synapse = child[str(e)][layer][i,o]
                        # Single point mutation on crossed-over genome:
                        child[str(e)][layer][i,o] = np.random.normal(synapse, math.fabs(synapse)) 

                self.children.append(child)

        else:

            while len(self.children) < self.popsize:
                # Randomly select and copy a parent
                parent_idx = random.choice(range(len(self.parents)))
                #while self.parents[parent_idx] == 0:
                #    parent_idx = random.choice(range(len(self.parents)))
                
                child = copy.copy(self.parents[parent_idx])

                if self.devo:
                    e = np.random.choice(range(self.environments+1))
                    d = np.random.choice(range(1, self.environments+1))
                    
                    child = self.Mutate(child, tensor='genome', e=e)
                    child = self.Mutate(child, tensor='dropout', e=d, l=0.0, u=1.0)
                else:
                    child = self.Mutate(child)

                self.children.append(child)


    def Mutate(self, content, tensor='genome', e=0, l=-1.0, u=1.0):
        # Mutate copied genome
        # Randomly select layer to affect
        IO = self.layer2layer[np.random.choice(range(len(self.layer2layer)))]
        layer = "{0}to{1}".format(IO[0], IO[1])
        i = np.random.choice(range(IO[0]))
        o = np.random.choice(range(IO[1]))
        # Mutate exisitng value at the above indices:
        synapse = content[str(e)][tensor][layer][i,o]
        content[str(e)][tensor][layer][i,o] = np.clip(np.random.normal(synapse, math.fabs(synapse)), l, u)
        return(content)

    def New_Swarm(self, E):

        self.children = []
        alpha = 0.1
        while len(self.children) < self.popsize:
            child = dict()
            for e in range(E):
                if self.dropout: 
                    child[str(e)] = {'genome':{}, 'dropout':{}}
                else:
                    child[str(e)] = {'genome':{}}
                for IO in self.layer2layer:
                    layer = "{0}to{1}".format(IO[0], IO[1])
                    if self.dropout:
                        # Remove dropout for base environment:
                        child['0']['dropout'] = {}
                        for tensor in child[str(e)].keys():
                            if e == 0 and tensor == 'dropout':
                                 next
                            else:
                                u_grad, s_grad, _mean, _std = self.Get_Particle_Data(IO, layer, e, tensor)
                                child[str(e)][tensor][layer] = Fill_Matrix(IO, _mean, _std, u_grad, s_grad, alpha)
                    else:
                        u_grad, s_grad, _mean, _std = self.Get_Particle_Data(IO, layer, e, 'genome')
                        child[str(e)]['genome'][layer] = Fill_Matrix(IO, _mean, _std, u_grad, s_grad, alpha)


            self.children.append(child)

    def Get_Particle_Data(self, IO, layer, e, tensor):
        
        u_grad = self.mean_grads[str(e)][layer]
        s_grad = self.mean_grads[str(e)][layer]
        _mean = self.means[str(e)][layer]
        _std = self.stds[str(e)][layer]
        return(u_grad, s_grad, _mean, _std)

