#!/usr/bin/env python2.7
# robot.py
# Shawn Beaulieu

from pyrosim import PYROSIM
import random
import constants as c
import numpy as np
import math

class ROBOT:
    def __init__(self, sim, genome, target_genome, blueprint, devo, gens, g):
        # call the method from the constructor the way you would in another script.
        # "SELF" is the generic name given as input to the constructor by the user
        # at a different time and location.
        self.offset = 0
        self.Send_Objects(sim)
        self.Send_Joints(sim)
        self.Send_Sensors(sim)
        if len(blueprint) > 1:
            self.Send_Deep_Neurons(sim, blueprint)
        else:
            self.Send_Neurons(sim, blueprint)
        self.Send_Synapses(sim, genome, target_genome, blueprint, devo, gens, g)
        #self.Send_Synapses(sim, genome, inflection, devo)

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

    def Send_Deep_Neurons(self, sim, blueprint):

        # ====== TODO: OPTIMIZE ======

        self.Sensor_to_Hidden(sim, blueprint[0])
        for b in blueprint[1:-1]:
            self.Hidden_to_Hidden(sim, b)
        self.Hidden_to_Motor(sim, blueprint[-1])

    def Hidden_to_Motor(self, sim, IO):       
        for i in range(IO[0]):
            sim.Send_Hidden_Neuron(neuronID=i+self.offset, tau=0.1)
        self.offset += IO[0]
        for o in range(IO[1]):
            sim.Send_Motor_Neuron(neuronID=o+self.offset, jointID=o, tau=0.3)

    def Sensor_to_Hidden(self, sim, IO):
        for i in range(IO[0]):
            sim.Send_Sensor_Neuron(neuronID=i, sensorID=i)
        self.offset += IO[0]
        for o in range(IO[1]):
            sim.Send_Hidden_Neuron(neuronID=o+self.offset, tau=0.1)
        self.offset += IO[1]

    def Hidden_to_Hidden(self, sim, IO):
        for i in range(IO[0]):
            sim.Send_Hidden_Neuron(neuronID=i+self.offset, tau=0.1)
        self.offset += IO[0]
        for o in range(IO[1]):
            sim.Send_Hidden_Neuron(neuronID=o+self.offset, tau=0.1)
        self.offset += IO[1]
       
    def Send_Neurons(self, sim, blueprint):
            for s in range(blueprint[0][0]):
                sim.Send_Sensor_Neuron(neuronID=s, sensorID=s)
            for m in range(blueprint[0][1]):
                sim.Send_Motor_Neuron(neuronID=m+blueprint[0][0], jointID=m, tau=0.3)

    def Send_Synapses(self, sim, genome, target_genome, dropout, blueprint, devo, gens, g):
        # Establish connection between sensor neurons and motor neurons
        devo_step = 1/gens
        #endTime = np.clip(1.0-devo_step*g, 0, 1)
        ID_tracker = 0 # Neuron ID
        for b in blueprint:
            ID_tracker += b[0]
            # layer index = "size of source layer"to"size of target layer"
            layer = "{0}to{1}".format(b[0], b[1])
            for I in range(b[0]):
                for O in range(b[1]):
                    if devo:
                        # Development schedule depends on proximity of base to target
                        #inflection = np.clip(np.sqrt((target_genome[layer] - genome[layer])**2),0,1)
                        # Create synapses (developmental)
                        sim.Send_Developing_Synapse(sourceNeuronID=I+ID_tracker-b[0], targetNeuronID=O+ID_tracker,
			    		startWeight=genome[layer][I,O], endWeight=target_genome[layer][I,O],
						dropTime=dropout[layer][I,O], startTime=0.0, endTime=1.0)
                    else:
                        # Create synapses (non-developmental)
                        sim.Send_Synapse(sourceNeuronID=I, targetNeuronID=O+ID_tracker, weight = genome[layer][I,O])
            ID_tracker += b[1]
