from pyrosim import PYROSIM
import matplotlib.pyplot as plt
import random
import constants as c
import numpy as np
import math

class ROBOT:
    def __init__(self, sim, genome, devo):
    #def __init__(self, sim, genome, epigenome, devo)
    #def __init__(self, sim, start_wts, end_wts, devo):
        # call the method from the constructor the way you would in another script.
        # "SELF" is the generic name given as input to the constructor by the user
        # at a different time and location.
        self.Send_Objects(sim)
        self.Send_Joints(sim)
        self.Send_Sensors(sim)
        self.Send_Neurons(sim)
	# Hadamard product between the genome and "epigenome"
	#wts = np.multiply(genome, epigenome)
  
        # Bayesian Evolution:
        #wts = np.zeros((13,13))
        #for row in range(13):
        #    for col in range(13):
                # random sample with mean = genome[row,col] and variance = 0.05
        #        wts[row,col] = 0.01*np.random.randn()+genome[row,col]

        self.Send_Synapses(sim, genome, devo)

    def Send_Objects(self, sim):
        sim.Send_Box(objectID =0, x=0, y =0, z=c.L + c.R, length =c.L, width=c.L, height=2*c.R, r=0.5,g=0.5,b=0.5)
        sim.Send_Cylinder(objectID=1, x=c.L, y=0, z= c.L+c.R, r1=1, r2=0, r3=0, length = c.L, radius = c.R, r=1, g=0, b=0.5)
        sim.Send_Cylinder(objectID=2, x=-c.L, y=0, z= c.L+c.R, r1=1, r2=0, r3=0, length = c.L, radius = c.R, r=1, g=0, b=0.5)
        sim.Send_Cylinder(objectID=3, x=0, y=c.L, z= c.L+c.R, r1=0, r2=1, r3=0, length = c.L, radius = c.R, r=1, g=0, b=0.5)
        sim.Send_Cylinder(objectID=4, x=0, y=-c.L, z= c.L+c.R, r1=0, r2=1, r3=0,length = c.L, radius = c.R, r=1, g=0, b=0.5)
        sim.Send_Cylinder(objectID=5, x=c.L+c.L/2, y=0, z=c.R+c.L/2,length = c.L,radius = c.R, r=1, g=0, b=0)
        sim.Send_Cylinder(objectID=6, x=-c.L-c.L/2, y=0, z=c.R+c.L/2,length = c.L,radius = c.R, r=1, g=0, b=0)
        sim.Send_Cylinder(objectID=7, x=0, y=c.L+c.L/2, z=c.R+c.L/2,length = c.L,radius = c.R, r=1, g=0, b=0)
        sim.Send_Cylinder(objectID=8, x=0, y=-c.L-c.L/2, z=c.R+c.L/2,length = c.L,radius = c.R, r=1, g=0, b=0)

    def Send_Joints(self, sim):
        sim.Send_Joint(jointID=0, firstObjectID = 0, secondObjectID = 3, x=0, y=c.L/2, z=c.L+c.R, n1=-1, n2=0, n3=0)
        sim.Send_Joint(jointID=1, firstObjectID = 3, secondObjectID = 7, x=0, y=c.L+c.L/2, z=c.L+c.R, n1=-1, n2=0, n3=0)
        sim.Send_Joint(jointID=2, firstObjectID = 0, secondObjectID = 4, x=0, y=-c.L/2, z=c.L+c.R, n1=-1, n2=0, n3=0)
        sim.Send_Joint(jointID=3, firstObjectID = 4, secondObjectID = 8, x=0, y=-c.L-c.L/2, z=c.L+c.R, n1=-1, n2=0, n3=0)
        sim.Send_Joint(jointID=4, firstObjectID = 0, secondObjectID = 1, x=c.L/2, y=0, z=c.L+c.R, n1=0, n2=-1, n3=0)
        sim.Send_Joint(jointID=5, firstObjectID = 1, secondObjectID = 5, x=c.L+c.L/2, y=0, z=c.L+c.R, n1=0, n2=-1, n3=0)
        sim.Send_Joint(jointID=6, firstObjectID = 0, secondObjectID = 2, x=-c.L/2, y=0, z=c.L+c.R, n1=0, n2=-1, n3=0)
        sim.Send_Joint(jointID=7, firstObjectID = 2, secondObjectID = 6, x=-c.L-c.L/2, y=0, z=c.L+c.R, n1=0, n2=-1, n3=0)

    def Send_Sensors(self, sim):
        sim.Send_Touch_Sensor(sensorID=0, objectID=7) #touch sensor inside object ID=0
        sim.Send_Touch_Sensor(sensorID=1, objectID=8)
        sim.Send_Touch_Sensor(sensorID=2, objectID=5) #touch sensor inside object ID=0
        sim.Send_Touch_Sensor(sensorID=3, objectID=6)
        #Light sensor, like position sensor, resides in the main body
        sim.Send_Light_Sensor(sensorID=4, objectID=0)
        #sim.Send_Position_Sensor(sensorID=10, objectID=0)

    def Send_Neurons(self, sim):
        # Ray sensor projects out into the y-axis
        for s in range(5):
            sim.Send_Sensor_Neuron(neuronID=s, sensorID=s)
        for m in range(8):
            sim.Send_Motor_Neuron(neuronID=m+5, jointID=m, tau=0.2)

    def Send_Synapses(self, sim, genome, targetGenome, devo=False):
    #def Send_Synapses(self, sim, start_wts, end_wts, devo=False):
        # Establish connection between sensor neurons and motor neurons

        for s in range(13):
            for m in range(13):
                if devo:
                    sim.Send_Developing_Synapse(sourceNeuronID=s,targetNeuronID=m,
					startWeight=genome[s,m], endWeight=targetGenome[s,m], startTime=0.0, endTime=random.uniform(0,1))
                #if devo:
                #    sim.Send_Developing_Synapse(sourceNeuronID=s,targetNeuronID=m,
		#			startWeight=start_wts[s,m],endWeight=end_wts[s,m],startTime=0,endTime=1.0)
                else:
                    sim.Send_Synapse(sourceNeuronID=s, targetNeuronID=m, weight = genome[s,m])

	#for s in range(5):
        #    for m in range(8):
        #        sim.Send_Synapse(sourceNeuronID=s, targetNeuronID=m+5, weight = wts[s,m])
        # see how far 'into' the screen our bots go. Use y-coordinate for this
