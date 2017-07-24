#!/usr/bin/env python
# Preprocess.py
# By Shawn Beaulieu
# July 21st, 2017

import numpy as np
import pandas as pd
import random
import ast

def sigmoid(x):
    return(1/(1+np.exp(-x)))

phenotypes = []                    
phenoFile1 = open("task1_HN_genomes.txt", "r")
for line in phenoFile1:
    phenotype = np.array(ast.literal_eval(line))
    phenotype = np.reshape(phenotype, [1,169]).flatten()
    phenotypes.append(sigmoid(phenotype))
phenoFile2 = open("task2_HN_genomes.txt", "r")
for line in phenoFile2:
    phenotype = np.array(ast.literal_eval(line))
    phenotype = np.reshape(phenotype, [1,169]).flatten()
    phenotypes.append(sigmoid(phenotype))

df_phenotypes = pd.DataFrame(phenotypes)
df_phenotypes = df_phenotypes.drop_duplicates()
df_phenotypes.to_csv("Processed_Phenotypes.csv", sep=",", header=None, index=None)

