#!/usr/bin/env python
# Run_HyperNEATBot.py
# Author: Shawn Beaulieu
# June 3rd, 2017

import sys, os
import numpy as np
from functools import partial
from peas.methods.neat import NEATPopulation, NEATGenotype
from peas.methods.hyperneat import HyperNEATDeveloper, Substrate
from peas.methods.reaction import ReactionDeveloper
from peas.networks.rnn import NeuralNetwork
from HyperNEATBot import HyperNEATBot

def evaluate(individual, task, developer):
    # First have to convert HyperNEAT genotype NEAT genotype
    # Then evaluate using HyperNEATBot.py Included in task
    # is the evaluate() function:

    stats = task.evaluate(developer.convert(individual))
    stats['nodes'] = len(individual.node_genes)
    #stats = task.evaluate(developer.convert(individual, 0), task_env='front')
    #stats_t2 = task.evaluate(developer.convert(individual, 1), task_env='back')
    #stats['fitness'] += stats_t2['fitness']
    #stats['fitness'] /= 2
    #stats['nodes'] = len(individual.node_genes)
   
    #if stats['fitness'] > 0.0:
            #with open("Compressed_genomes.txt", "a+") as genomefile:
            #    genomefile.write(str(developer.convert(individual,1).cm.tolist()))
            #    genomefile.write("\n")
            #with open("Compressed_fitness.txt", "a+") as fitfile:
            #    fitfile.write(str(stats['fitness']))
            #    fitfile.write("\n")
            #with open("Compressed_fitness_t1.txt", "a+") as evidence_t1:
            #    t1_fitness = (stats['fitness']*2) - stats_t2['fitness']
            #    evidence_t1.write(str(t1_fitness))
            #    evidence_t1.write("\n")
            #with open("Compressed_fitness_t2.txt", "a+") as evidence_t2:
            #    t2_fitness = stats_t2['fitness']
            #    evidence_t2.write(str(t2_fitness))
            #    evidence_t2.write("\n")

    return(stats)

def solve(individual, task, developer):
    """
    Checks if a given robot satisfies the criteria for solving a task:
    
    """
    return(task.solve(developer.convert(individual)))
    #task1 = task.solve(developer.convert(individual, 0), task_env='front')
    #task2 = task.solve(developer.convert(individual, 1), task_env='back')
    #isSolved = np.array([task1, task2])
    #return(float((isSolved > 1.0).all()))
    
def run(generations, popsize, prob_add_conn, prob_add_node, types):
    task = HyperNEATBot()
    # Substrate is a (nodes x nodes) matrix: We've 5 sensors and 8 motors,
    # totaling 13:
    substrate = Substrate(nodes_or_shape=(13,1))
    # Weight range -1.0 to 1.0 to excitatory and inhibitory input
    geno = lambda: NEATGenotype(inputs=5, outputs=8, weight_range=(-1.0, 1.0), 
                                prob_add_conn=prob_add_conn,prob_add_node=prob_add_node, 
                                types=types)
    pop = NEATPopulation(geno, popsize=popsize)

    developer = HyperNEATDeveloper(substrate=substrate, add_deltas=False, sandwich=False)

    # Use partial to call evaluate() using task and developer.
    # In "evaluator" we pass each individual to the partial function for evaluator, which calls
    # the function evaluate() for said individual with the parameters "task" and "developer"
    results = pop.epoch(generations=generations,evaluator=partial(evaluate, task=task, developer=developer),
                        solution=partial(solve,task=task,developer=developer))
    
if __name__ == '__main__':
    types=['sin','tanh','linear','exp','abs','gauss']
    generations = 200
    popsize = 75
    prob_add_conn = 0.3
    prob_add_node = 0.03

    #genomes = {}
    #for task in range(1,3):
    #    fitness_file = open("task{0}_HN_fitness.txt".format(task))
    #    genome_file = open("task{0}_HN_genomes.txt".format(task))
    #    genome = []
    #    fitness_values = []
    #    for line in fitness_file:
    #        fitness_values.append(float(line))
    #    fittest_idx = np.argmax(fitness_values)
    #    for line in genome_file:
    #        genome.append(line)
    #    genomes[str(task)] = np.array(ast.literal_eval(genome[fittest_idx]))

    run(generations, popsize, prob_add_conn, prob_add_node, types)
