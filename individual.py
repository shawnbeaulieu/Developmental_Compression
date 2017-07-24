from pyrosim import PYROSIM
import matplotlib.pyplot as plt
import random
import math
import numpy as np
import constants as c
from robot import ROBOT

class INDIVIDUAL:
	def __init__(self, genome, devo):
        #def __init__(self, genome, epigenome, devo):

		# Compression: Two objectives (1) maximize fitness in both
		# environments; (2) minimize difference between front and
		# back genomes
		self.genome = genome
		#self.targetGenome = targetGenome
                self.devo = devo
                #self.fitness_T1 = 0
		#self.fitness_T2 = 0
		self.fitness = 0
                self.env_tracker = None

                # self.env_tracker keeps track of what environment the robot is in
                # (e.g. front or back)

	def Start_Evaluation(self, env, pp, pb, env_tracker, ipy=False):

                #Want to send the environment to the simulator just after robot

		self.env_tracker = env_tracker
                self.sim = PYROSIM(playPaused=pp, evalTime=1000, debug=False, playBlind=pb)
		robot = ROBOT(self.sim, self.genome, self.devo)
                #robot = ROBOT(self.sim, self.genome, self.targetGenome, self.devo)
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

                    'Joint_0': [0],
                    'Joint_1': [0],
                    'Joint_2': [0],
                    'Joint_3': [0],
                    'Joint_4': [0],
                    'Joint_5': [0],
                    'Joint_6': [0],
                    'Joint_7': [0]

		}
                
               
                # FITNESS:

                self.fitness = self.Sensor_Data['zLight_Sensor'][-1]

                #if self.env_tracker == 0:
                #    self.fitness_T1 += self.Sensor_Data['zLight_Sensor'][-1]
                #else:
                #    self.fitness_T2 += self.Sensor_Data['zLight_Sensor'][-1]
                
                #if self.fitness >= 0.20:
                #    sensor_matrix = self.Sensor_Data['Touch_Sensor0'][:, np.newaxis]
                #    sensor_idx = sorted(self.Sensor_Data.keys())
                #    for s in range(1,5):
                #        sensor_matrix = np.concatenate((sensor_matrix, self.Sensor_Data[sensor_idx[s]][:, np.newaxis]), axis=1)
                        #   S0     S1    S2    S3    S4    LIGHT
                        #    .      .     .     .     .      .
                        #    .      .     .     .     .      .
                        #    .      .     .     .     .      .
                #    with open("Sensor_Data_bothTasks.txt", "a+") as sensorFile:
                #        sensorFile.write(str(sensor_matrix.tolist()))
                #        sensorFile.write("\n")

                        # Now, MOTOR DATA:
                        #   M0     M1     M2  ....
                        #    .      .      .
                        #    .      .      .
                        #    .      .      .
                #    self.Obtain_Motor_Data()

                #    motor_idx = sorted(self.Motor_Data.keys())
                #    motor_matrix = self.Motor_Data['Joint_0'][:, np.newaxis]
                #    for m in range(1,8):
                #        motor_matrix = np.concatenate((motor_matrix, self.Motor_Data[motor_idx[m]][:, np.newaxis]), axis=1)
                #    with open("Motor_Data_bothTasks.txt", "a+") as motorFile:
                #        motorFile.write(str(motor_matrix.tolist()))
                #        motorFile.write("\n")

                #self.fitness += Sensor_Data['zLight_Sensor'][-1]
                
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
                    # Convert each element into an array:
                self.Motor_Data[sorted(self.Motor_Data.keys())[motor]] = np.array(self.Motor_Data[sorted(self.Motor_Data.keys())[motor]])

	def Print_Fitness(self):

		#T1 = self.fitness_T1
		#T2 = self.fitness_T2
		#if T2 >= 0.80*T1 and T1 >= 0.80*T2:
		#    self.fitness = ((T1+T2)/2)
		#else:
		#    self.fitness = (min(T1,T2))
                #self.fitness = (T1+T2)/2

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

