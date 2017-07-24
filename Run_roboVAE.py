#!/usr/bin/env python
# Run_roboVAE.py
# By Shawn Beaulieu
# July 21st, 2017

import ast
import math
import random
import numpy as np
import pandas as pd
from roboVAE import VariationalAutoEncoder
from pyrosim import PYROSIM
from environments import ENVIRONMENTS
from individual import INDIVIDUAL
from robot import ROBOT

def bernoulli2weight(x):
    weights = (-np.log(1e-10 + (1/x+1e-10) - 1.0))
    np.clip(weights, a_min=-1.0, a_max=1.0)
    return(weights)

def generate_batch(X, batch_size):
    num_instances = X.shape[0]
    randSample = random.sample(range(num_instances), batch_size)
    batch = X[randSample, :]
    return(batch)

def Train(data, blueprint, learning_rate, batch_size, training_epochs, display_step):

    VAE = VariationalAutoEncoder(blueprint, learning_rate=learning_rate, batch_size=batch_size)
    for epoch in range(training_epochs):
        cost = 0.0
        num_batches = int(len(data)/(batch_size))
        for iteration in range(num_batches):
            X_batch = generate_batch(data, batch_size)
            cost += VAE.Fit(X_batch)/batch_size
        cost /= num_batches
        if epoch % display_step == 0:
            print("Epoch {0}: Cost = {1}".format(epoch, cost))

    return(VAE)

if __name__ == '__main__':

    phenotypes = np.array(pd.read_csv("Processed_Phenotypes.csv", sep=",", header=None))
    blueprint = {

        'input_dim': 169,
        'h1': 500,
        'h2': 400,
        'z_dim': 25,
        'h3': 400,
        'h4': 500
 
    }
    batch_size = 100
    training_epochs = 500
    VAE = Train(phenotypes, blueprint, learning_rate=0.001, batch_size=batch_size, training_epochs=training_epochs, display_step=1)

    # After training generate a new phenotype and evaluate its performance in simulation.
    newPhenotype = VAE.Generate()[random.sample(range(batch_size), 10)]
    newPhenotype = bernoulli2weight(newPhenotype)
    for sample in range(10):
        phenotype = np.reshape(newPhenotype[sample, :], [13,13])
        milieu = ENVIRONMENTS()
        agent = INDIVIDUAL(phenotype, devo=False)
        for task in range(2):
            agent.Start_Evaluation(milieu.envs[task], pp=True, pb=False, env_tracker=task)
            agent.Compute_Fitness()
            fitness = agent.Print_Fitness()
            print(fitness)

