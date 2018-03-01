#!/usr/bin/env python2.7
# individual.py
# Shawn Beaulieu

from pyrosim import PYROSIM
import random
import math
import numpy as np
import pandas as pd
import constants as c
from robot import ROBOT

class INDIVIDUAL:
        def __init__(self, genome, target_genome, blueprint, devo, gens, g):

                self.genome = genome
		self.target_genome = target_genome
                self.devo = devo
                self.blueprint = blueprint
                self.gens = gens
                self.g = g
		self.fitness = 0
                self.env_tracker = None

                # self.env_tracker keeps track of what environment the robot is in
                # (e.g. front or back)

	def Start_Evaluation(self, env, pp, pb, env_tracker, ipy=False):

                #Want to send the environment to the simulator just after robot

		self.env_tracker = env_tracker
                self.sim = PYROSIM(playPaused=pp, evalTime=1000, debug=False, playBlind=pb)
		robot = ROBOT(self.sim, self.genome, self.target_genome, self.blueprint, self.devo, self.gens, self.g)
                env.Send_To(self.sim)
                self.sim.Start()
                # For running program in ipython, set ipy=True
                if ipy:
                    self.sim.Wait_To_Finish()

	def Compute_Fitness(self):

        	self.sim.Wait_To_Finish() #says "pause until simulation is finished"
		
		# Lehman Mutual Information:
		#Average_Sensor_Values = np.zeros(c.evalTime)
                
		self.Sensor_Data = {

                    'Touch_Sensor0': np.array(self.sim.Get_Sensor_Data(sensorID=0)),
		    'Touch_Sensor1': np.array(self.sim.Get_Sensor_Data(sensorID=1)),
		    'Touch_Sensor2': np.array(self.sim.Get_Sensor_Data(sensorID=2)),
		    'Touch_Sensor3': np.array(self.sim.Get_Sensor_Data(sensorID=3)),
		    'zLight_Sensor': np.array(self.sim.Get_Sensor_Data(sensorID = 4))
                    # 'z' so it appears lasted in sorted()
		}
                
                self.Motor_Data = {

                    'Joint_0': [],
                    'Joint_1': [],
                    'Joint_2': [],
                    'Joint_3': [],
                    'Joint_4': [],
                    'Joint_5': [],
                    'Joint_6': [],
                    'Joint_7': []

		}
                
               
                # FITNESS:
                #print(self.sim.dataFromPython)
                self.fitness = sum(self.Sensor_Data['zLight_Sensor'])/len(self.Sensor_Data['zLight_Sensor'])
                #self.fitness = self.Sensor_Data['zLight_Sensor'][-1]
                #if self.env_tracker == 0:
                #    self.fitness_T1 += self.Sensor_Data['zLight_Sensor'][-1]
                #else:
                #    self.fitness_T2 += self.Sensor_Data['zLight_Sensor'][-1]
                
                #directory = "/users/s/b/sbeaulie/robotics/pyrosim/"
                #sensor_matrix = self.Sensor_Data['Touch_Sensor0'].reshape(-1,1)
                #sensor_idx = sorted(self.Sensor_Data.keys())
                #for s in range(1,5):
                #    new_data = self.Sensor_Data[sensor_idx[s]].reshape(-1,1)
                #    sensor_matrix = np.concatenate([sensor_matrix, new_data], axis=1)
                        #   S0     S1    S2    S3    S4    LIGHT
                        #    .      .     .     .     .      .
                        #    .      .     .     .     .      .
                        #    .      .     .     .     .      .
                #sensor_df = pd.DataFrame(sensor_matrix)
                #try:
                #    sensor_df.to_csv("{0}Sensor_Data_T{1}.csv".format(directory, self.env_tracker + 1), mode="a", sep=",", header=None, index=None)
                #except:
                #    sensor_df.to_csv("{0}Sensor_Data_T{1}.csv".format(directory, self.env_tracker + 1), sep=",", header=None, index=None)
                        # Now, MOTOR DATA:
                        #   M0     M1     M2  ....
                        #    .      .      .
                        #    .      .      .
                        #    .      .      .
                #self.Obtain_Motor_Data()
                #motor_idx = sorted(self.Motor_Data.keys())
                #motor_matrix = np.array(self.Motor_Data['Joint_0']).reshape(-1,1)
                #for m in range(1,8):
                #    new_data = np.array(self.Motor_Data[motor_idx[m]]).reshape(-1,1)
                #    motor_matrix = np.concatenate([motor_matrix, new_data], axis=1)
                #motor_df = pd.DataFrame(motor_matrix)
                #print(motor_df)
                #try:
                #    motor_df.to_csv("{0}Motor_Data_T{1}.csv".format(directory, self.env_tracker + 1), mode="a", sep=",", header=None, index=None)
                #except:
                #    motor_df.to_csv("{0}Motor_Data_T{1}.csv".format(directory, self.env_tracker + 1), sep=",", header=None, index=None)
                
        def Obtain_Motor_Data(self):

            for motor in range(8):
                for time in range(len(self.Sensor_Data['Touch_Sensor0'])):
                    summed_input = 0
                    for sensor in range(5):
                        current_sensor = self.Sensor_Data[sorted(self.Sensor_Data.keys())[sensor]]
                        summed_input += self.genome[sensor, motor+5]*current_sensor[time]

                    current_motor = self.Motor_Data[sorted(self.Motor_Data.keys())[motor]]
                    # Following equation for motor output, each new output depends on previous state
                    # for "momentum"
                    if time == 0:
                        current_motor.append(np.tanh(summed_input))
                    else:
                        current_motor.append(np.tanh(current_motor[time-1] + summed_input))
	
        def Print_Fitness(self):

		#T1 = self.fitness_T1
		#T2 = self.fitness_T2
		#if T2 >= 0.80*T1 and T1 >= 0.80*T2:
		#    self.fitness = ((T1+T2)/2)
		#else:
		#    self.fitness = (min(T1,T2))
                #self.fitness = (T1+T2)/2
                #print(self.fitness)
                return(self.fitness)

	def Mutate(self):

                i_node = random.randint(0,4)
                o_node = random.randint(0,7)
		# Gaussian has mean = self.genome[geneToMutate] and variance math.fabs...
		# both are defined by the randomly selected gene that is being mutated.
		# i.e. mean and variance are determined by the value of a single "gene".
		self.genome[i_node, o_node] = random.gauss(self.genome[i_node,o_node], math.fabs(self.genome[i_node,o_node]))
	
	def Print(self):
		#print("[{0},{1},{2},{3}]").format(self.ID, self.fitness, self.genome_ih, self.genome_ho)
                print("[{0},{1}]").format(self.ID, self.objective_one)

