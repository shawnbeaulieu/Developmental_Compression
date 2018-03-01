import glob
import pandas as pd
import numpy as np
from pyrosim import PYROSIM                
from robot import ROBOT
from individual import INDIVIDUAL        
from environments import ENVIRONMENTS

milieu = ENVIRONMENTS()
for t in range(0,2):
    files = glob.glob("HN_Weights_T{0}*.csv".format(t+1))
    counter = 0    
    for f in files:
        genomes = pd.read_csv(f, header=None)
        for i in range(genomes.shape[0]):
            if counter > 500:
                break
            else:
                w = np.array(genomes.iloc[i,:-1]).reshape(13,13)
                agent = INDIVIDUAL(w, target_genome=0, inflection=0.0, devo=False)
                agent.Start_Evaluation(milieu.envs[t], pp=False, pb=True, env_tracker=t)
                agent.Compute_Fitness()
                counter += 1

