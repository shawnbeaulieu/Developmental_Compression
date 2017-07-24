import constants as c
from environment import ENVIRONMENT

class ENVIRONMENTS:
	
	def __init__(self):
		self.envs = {}
		for i in range(c.numEnvs):
			self.envs[i] = ENVIRONMENT(i)
