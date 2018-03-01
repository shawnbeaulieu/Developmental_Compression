#!/usr/bin/env python2.7
# environment.py
# Shawn Beaulieu

import random
from pyrosim import PYROSIM
import constants as c

class ENVIRONMENT:
	
	def __init__(self, ID):
		#The following code (self.I, etc.) stores the size of the box in the
		#environment, as well as its position. Length, width, and height
		#set to the valye of the robot's leg length		
		self.l = 2*c.L 
		self.w = 2*c.L
		self.h = 2*c.L
		self.x = 0
		self.y = 0
		self.z = 0.05
                if ID == 0:		
			self.Place_Light_Source_To_The_Front()
 		if ID == 1:
			self.Place_Light_Source_To_The_Back()
                if ID == 2:
                        self.Place_Light_Source_To_The_Right()
                if ID == 3: 
                        self.Place_Light_Source_To_The_Left()
		#print(self.l, self.w, self.h, self.x, self.y, self.z)
	
	def Place_Light_Source_To_The_Front(self):
			
		self.I = 2*c.L
                self.w = 2*c.L
                self.h = 2*c.L
                self.x = 0
                self.y = 30*c.L
                self.z = 0.05

	def Place_Light_Source_To_The_Right(self):

                self.I = c.L
                self.w = c.L
                self.h = c.L
                self.x = 30*c.L
                self.y = 0
                self.z = 0.05
	
	def Place_Light_Source_To_The_Back(self):

                self.I = 2*c.L
                self.w = 2*c.L
                self.h = 2*c.L
                self.x = 0.0
                self.y = -30*c.L
                self.z = 0.05

	def Place_Light_Source_To_The_Left(self):

                self.I = c.L
                self.w = c.L
                self.h = c.L
                self.x = -30*c.L
                self.y = 0
                self.z = 0.05

	def Send_To(self, sim):
		sim.Send_Box(objectID = 9, x=self.x, y=self.y,z=self.z, \
			     length=self.l,width=self.w,height=self.h)
		sim.Send_Light_Source(objectIndex = 9)
