#!/usr/bin/env python
# roboVAE.py
# By Shawn Beaulieu
# July 21st, 2017

"""
For use in Python 2.7

If python 2 and 3 are both installed on your system, use
"source python2" prior to running this code. Python 2.7 is required
for running Pyrosim. Later versions of Python are currently incompatible.

Implementation of a conditional variational autoencoder (VAE) in tensorflow.
The input data are the phenotypic weight matrices for robots evolved
using HyperNEAT. Inspiration for this code taken from the following resources:

THEORY:

(1) https://arxiv.org/pdf/1606.05908.pdf
(2) https://arxiv.org/pdf/1601.00670.pdf

CODE:

(1) http://blog.fastforwardlabs.com/2016/08/22/under-the-hood-of-the-variational-autoencoder-in.html
(2) https://jmetzen.github.io/2015-11-27/vae.html
(3) Tensorflow tutorials via Tensorflow

Summary:

The purpose of a variational autoencoder (VAE) is to perform inference on  the input data, X,
so as to identify the latent features on which X is predicated (e.g. what are the (approximately) 
'irreducible' features that causally produce X?). This is useful for generating synthetic data that 
closely resembles the input data on which the network (encoder) was trained. Training consists of two 
parts: (i) using the compressed latent code, try to generate an instance that closely resembles the 
training data; (ii) backpropagate errors in reconstruction so as to better reproduce the input.

Essentially, the architecture is composed of two networks: an encoder and a decoder, both of
which are neural networks. A technique called reparameterization is used in sampling the latent representation
as sampling methods aren't ammenable to backpropagation. In practice, this means that we first generate samples
from a standard normal Gaussian distribution e ~ N(0,I), which is then used to modify the latent space as follows:
z = mean + var*e, where the multiplication here is a Hadamard product between the variance and the sampled value,
epsilon.

ENCODER: q(z | x) parameterized by phi
DECODER: p(x | z) parameterized by theta

Where phi and theta refer to the parameters of the respective neural networks.

Formally, we are maximizing the expected log probability of the data given the latent representation with 
respect to the latent space, and a penalty term (KL divergence) that constrains our approximation 
of the posterior to be close to our initial prior, N(0,I) over the latent space. The motivation for this is 
to prevent the encoder from assigning a unique point in the latent space to each input instance. We want the 
encoder to be frugal in its use of the latent space for compact representation.

All of this has the effect of maximizing the lower bound on the log probability of the data, p(X).

For more information, see the aforementioned links covering the theory of VAEs.


"""

import math
import random
import functools
import tensorflow as tf
import numpy as np
from pyrosim import PYROSIM
from robot import ROBOT
from environments import ENVIRONMENTS
from individual import INDIVIDUAL
from functional import partial, compose

def bernoulli2weight(x):
    weights = (-np.log((1.0/x) - 1.0))
    np.clip(weights, a_min=-1.0, a_max=1.0)
    return(weights)

def Xavier(name, shape):
    """
    To guard against both vanishing and exploding gradients. The variance
    of the distribution from which we draw random samples for weights
    is a function of the number of input neurons for a given layer
    (and for the case of Bengio initialization, the output neurons as well) 

    INPUTS:
    name: string containing exact name of layer being initialized
    shape: dimensions of the weight matrix: (e.g. (input,output)) 

    """
    return(tf.get_variable(name, shape=shape, initializer=tf.contrib.layers.xavier_initializer()))

def Recursion(*args):
    """
    Creates a sequence of nested functions.
    - *args: a variable-length input to the function
    - functools.reduce: applies compose() to each argument in args
    - compose: strings together functions, defining a recursive function
    - partial: the returned function, which accepts a value, X, that is then
      passed through the recursive function

    >> Encoder = Recursion([f,g,z])
    >> Encoder(x) = f(g(z(x)))

    """
    
    return(partial(functools.reduce, compose)(*args))

class VariationalAutoEncoder(object):
    def __init__(self, blueprint, activation=tf.nn.elu, learning_rate=0.001, batch_size=100):
        """ 
        INPUTS:

        blueprint: a dictionary containing the size (number of neurons) for each layer
        activation: desired activation function for the network
        learning_rate: the magnitude with which changes to the network's weights are made
        batch_size: SGD is used for speed of convergence. Lower variance results from larger batches

        blueprint = {

        'input_dim': 169,     
        'h1': 300,
        'h2': 100,
        'z_dim' = 25,
        'h3': 100,
        'h4': 300
  
        }

        """
           
        self.blueprint = blueprint
        self.activation = activation
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.simulation_loss = None
        # self.x is initialized as a placeholder of dimensions (? x input_features)
        # This is so that the batch size can vary dynamically, rather than be hardcoded:
        self.x = tf.placeholder(tf.float32, [None, blueprint['input_dim']])
        self.Parameterize()
        self.Compose_Network()
        self.Optimize_Objective()
        
        # Initialize tensorflow variables. Computational graph doesn't exist
        # until the variables are initialized. Said variables declared both above
        # (explicitly in the case of self.x) and in the previously called functions:
        build = tf.global_variables_initializer()
        self.sess = tf.InteractiveSession()
        self.sess.run(build)
    
    def Parameterize(self):
        """
        Creates the weights and biases for the network using
        Xavier initialization for the weights of each layer, and
        initialization to 0 for all biases.

        """

        weights = dict()
        biases = dict()

        weights['encoder'] = {
            'inputTOh1': Xavier('inputTOh1', (self.blueprint['input_dim'], self.blueprint['h1'])),
            'h1TOh2': Xavier('h1TOh2', (self.blueprint['h1'], self.blueprint['h2'])),
            'h2TOz_mean': Xavier('h2TOz_mean', (self.blueprint['h2'], self.blueprint['z_dim'])),
            'h2TOz_log_var': Xavier('h2TOz_log_var', (self.blueprint['h2'], self.blueprint['z_dim']))
        }

        biases['encoder'] = {
            'inputTOh1': tf.Variable(tf.zeros([self.blueprint['h1']], dtype=tf.float32)),
            'h1TOh2': tf.Variable(tf.zeros([self.blueprint['h2']], dtype=tf.float32)),
            'h2TOz_mean': tf.Variable(tf.zeros([self.blueprint['z_dim']], dtype=tf.float32)),
            'h2TOz_log_var': tf.Variable(tf.zeros([self.blueprint['z_dim']], dtype=tf.float32))
        }

        weights['decoder'] = {
            'zTOh3': Xavier('zTOh3', (self.blueprint['z_dim'], self.blueprint['h3'])),
            'h3TOh4': Xavier('h3TOh4', (self.blueprint['h3'], self.blueprint['h4'])),
            'h4TOoutput_mean': Xavier('h4TOoutput_mean', (self.blueprint['h4'], self.blueprint['input_dim']))
        }

        biases['decoder'] = {
            'zTOh3': tf.Variable(tf.zeros([self.blueprint['h3']], dtype=tf.float32)),
            'h3TOh4': tf.Variable(tf.zeros([self.blueprint['h4']], dtype=tf.float32)),
            'h4TOoutput_mean': tf.Variable(tf.zeros([self.blueprint['input_dim']], dtype=tf.float32))
        }

        self.weights = weights
        self.biases = biases
    
    def Encode(self):
        """
        Passes the input data, self.x, through the first two hidden layers and into
        the latent layer, which encodes for the mean and variance of the variational 
        distribution, Q, from which we layer sample in the decoding phase.

        """
        weights = self.weights['encoder']
        biases = self.biases['encoder']

        # Create tensorflow graph through which we'll pass phenotypic data:
        # Just a series of geometric transformations: X => h1 => h2 => z
        layer_one = self.activation(tf.add(tf.matmul(self.x, weights['inputTOh1']), biases['inputTOh1']))
        dropout_1 = tf.layers.dropout(layer_one, rate=0.0)
        layer_two = self.activation(tf.add(tf.matmul(dropout_1, weights['h1TOh2']), biases['h1TOh2']))
        dropout_2 = tf.layers.dropout(layer_two, rate=0.0)
        z_mean = tf.add(tf.matmul(dropout_2, weights['h2TOz_mean']), biases['h2TOz_mean'])
        z_log_var = tf.add(tf.matmul(dropout_2, weights['h2TOz_log_var']), biases['h2TOz_log_var'])
        
        # Being that this is variational inference, which is a Bayesian inference method, the output of the
        # encoder is necessarily probabilistic. Hence, we return the parameters for a Gaussian distribution
        # corresponding to our approximation of the true posterior, q(z | x).
  
        return(z_mean, z_log_var)

    def Decode(self):

        """
        Takes compressed data from the latent space and reconstructs
        the data by passing it through a decoder network. The reparameterization
        trick is used, wherein Gaussian noise, N(0,1), is added to the latent
        representation so that backpropagation can flow through the parameters
        that govern the variational distribution. 

        Recall that in VAEs, the output is represented as a distribution over the
        feature space. The weights here aren't Bernoulli distributed, as is the case
        for the MNIST dataset, but range from [-1.0, 1.0]. For the purposes of creating a
        consistent loss function, we first convert to a Bernoulli distribution, then
        during simulation we move back into the range [-1.0, 1.0]
       
        """
        weights = self.weights['decoder']
        biases = self.biases['decoder']

        layer_three = self.activation(tf.add(tf.matmul(self.z, weights['zTOh3']), biases['zTOh3']))
        dropout_3 = tf.layers.dropout(layer_three, rate=0.30)
        layer_four = self.activation(tf.add(tf.matmul(dropout_3, weights['h3TOh4']), biases['h3TOh4']))
        dropout_4 = tf.layers.dropout(layer_four, rate=0.30)
        output_mean = tf.nn.sigmoid(tf.add(tf.matmul(dropout_4, weights['h4TOoutput_mean']), biases['h4TOoutput_mean']))
        
        return(output_mean)

    def Compose_Network(self):
        # Encode input (probabilistically):
        self.z_mean, self.z_log_var = self.Encode()
        # Sample from N(0,I):
        epsilon = tf.random_normal((self.batch_size, self.blueprint['z_dim']), mean=0.0, stddev=1, dtype=tf.float32)
        # Reparameterization:
        self.z = tf.add(self.z_mean, tf.multiply(tf.sqrt(tf.exp(self.z_log_var)), epsilon))
        # Decode latent representation (probabilistically):
        self.output_mean = self.Decode()

    def Reconstruction_Loss(self):
        # Canonical Bernoulli: (h^(x))*((1-h)^(1-x))
        # Log loss: x*(log(h)) + (1-x)(log(1-h))
        # Intuitively, this negative log probability can be envisaged as the nats required to reconstruct the
        # input.
        return(-tf.reduce_sum(self.x*(tf.log(1e-10 + self.output_mean)) + (1-self.x)*(tf.log(1e-10 + 1 - self.output_mean)), \
                axis=1))

    def KL_Divergence(self):
        return(-0.5*(tf.reduce_sum(1 + self.z_log_var - tf.square(self.z_mean) - tf.exp(self.z_log_var), axis=1)))

    def Optimize_Objective(self):
        """ 
        The output of the decoder should both resemble the input and achieve comparable performance
        on the given task.

        """
        # Two terms in the loss function: E[log(P(x | z))] - KL[Q(z | x)||P(z)] 
        # (i) reconstruction loss = E[log(P(x | z))]
        # (ii) latent loss = KL[Q(z | x)||P(z)]
        # Has closed form solution (see notes) given that both terms
        # are (multivariate) Gaussians.

        # Weight the loss function by the relative fitness of the newly constructed phenotype.
        # Because fitness values range from [0,1] this weight will serve the function of ensuring
        # that the hypothesized reconstruction has a similar degree of fitness when placed in the
        # appropriate environment.

        if self.simulation_loss == None:
            self.cost = tf.reduce_mean(self.Reconstruction_Loss() + self.KL_Divergence())
        else:
            self.cost = tf.reduce_mean(self.Reconstruction_Loss() + self.KL_Divergence())*self.simulation_loss
        
        # GOAL: minimize the negative log likelihood as much as possible. Output of loss should
        # be INCREASINGLY negative as epochs go by.

        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.cost)

    def Simulate(self, real, synthetic):

        real_phenotype = bernoulli2weight(real)
        real_phenotype = np.reshape(real_phenotype, [13,13])
        synthetic_phenotype = bernoulli2weight(synthetic)
        synthetic_phenotype = np.reshape(synthetic_phenotype, [13,13])

        milieu = ENVIRONMENTS()

        real_agent = INDIVIDUAL(real_phenotype, devo=False)
        real_agent.Start_Evaluation(milieu.envs[0], pp=False, pb=True, env_tracker=0)
        real_agent.Compute_Fitness()
        real_fitness = real_agent.Print_Fitness()

        synthetic_agent = INDIVIDUAL(synthetic_phenotype, devo=False)
        synthetic_agent.Start_Evaluation(milieu.envs[0], pp=False, pb=True, env_tracker=0)
        synthetic_agent.Compute_Fitness()
        synthetic_fitness = synthetic_agent.Print_Fitness()

        # Simulation loss is constructed such that the difference between the real_fitness and 
        # synthetic fitness ranges between [0,1], yielding low numbers for synthetic phenotypes
        # that closely mirror real phenotypes and high numbers for those that diverge.
        # +1 is added so that when we minimize the total cost we seek reconstructions whose
        # simulation weight = 1 (which will have no effect on the total cost).
        # e.g. self.cost = (-1.6 + (-2.3))*1.30, resulting in larger penalty than
        # (-1.6 + (-2.3))*1.10 for a phenotype that achieves high fitness.

        self.simulation_loss = 1 + (1 - (real_fitness - synthetic_fitness))

    def Fit(self, X):
        """
        For use in training the model.        

        """
        #z_mean = self.Map(X)
        #output_mean = self.Generate(z_mean)        
        #self.Simulate(X, output_mean)
        opt, cost = self.sess.run((self.optimizer, self.cost), feed_dict={self.x: X})
        return(cost)
    
    def Map(self, X):
        """
        Map the input data to the latent space by running self.sess.run(self.z_mean...)
        This evaluates the tensor self.z_mean for the TF graph created using the functions above

        """
        return(self.sess.run(self.z_mean, feed_dict={self.x: X}))

    def Generate(self, z_mean=None):
       
        """
        Probe the latent space, Z, or if Z hasn't yet been computed, generate a
        random sample from a unit Gaussian of size z_dim. Then evaluate the tensor
        self.output_mean using the aforementioned sample.

        """
        # As per the VAE tutorial, we sample from the standard normal distribution N(0,I)
        # to generate new samples:
        if z_mean == None:
            z_mean = np.random.normal(size=(self.batch_size,self.blueprint['z_dim']), loc=0, scale=1.0)

        return(self.sess.run(self.output_mean, feed_dict={self.z: z_mean}))

    def Reconstruct(self, X):
        """
        Full pass through the network, from input to hypothesized/reconstructed
        output.
        
        """
        return(self.sess.run(self.output_mean, feed_dict={self.x: X}))
