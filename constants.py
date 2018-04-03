#Evaluation time is the default if none is specified.


#simulator constants
evaluation_time = 1;
dt = 0.05;
gravity = -0.5;
hpr=[121,-27.5000,0.0000];
xyz=[0.8317,-0.9817,0.8000];

#robot constants
#Adding two variables: length and radiius of robot's leg segments

L = 0.1 #For the construction of robots
R = L/5 #See above
evalTime = 1000
popSize = 10
numGens = 100
numEnvs = 4
