#!/usr/bin/env python
# taskNet.py
# By Shawn Beaulieu
# July 8th, 2017

"""
A recurrent neural network for classification built using Keras
and a Tensorflow backend (also works with Theano). LSTM cells are
used due mainly to their popularity, but GRUs can be substituted, as
they exhibit comparable performance with fewer parameters.

Intended for classifying received sensor data according to which task
generated it in Pyrosim. Although the code is general enough to be adapted
for other purposes in sequential classification.

TODO: Enhance modularity. Allow user to specify architecture (e.g. stacked
LSTM/GRUs, number of Dense layers, embeddings, etc.)

EXAMPLE USAGE:

>> from taskNet import RNN

>> model = RNN(myData, myLabels, class_size=2)
>> model.Construct_LSTM()
>> model.Fit()
>> model.Evaluate()
Accuracy = 85.60%

"""

import random
import math
import numpy as np
from keras.models import Sequential
from keras.layers import Flatten, Dropout, Dense, LSTM

class RNN:
    def __init__(self, X, y, num_classes=2, batch_size=20):

        """
        Define instance variables. Number of classes and batch size
        can both be adjusted depending on the task. Data is shuffled
        and divided into training and test sets in Construct_LSTM()

        """
        # Ensure that data is shuffled:
        self.batch_size = batch_size
        self.num_classes = num_classes
        self.num_instances = X.shape[0]
        self.timesteps = X.shape[1]
        self.dimensionality = X.shape[2]
        shuffled_idx = np.random.permutation(range(self.num_instances))
        self.Source = X[shuffled_idx, :, :]
        self.Target = y[shuffled_idx]

    def Construct_LSTM(self, train_size=0.80, objective='sparse_categorical_crossentropy', optimizer='rmsprop'):

        """
        Build the LSTM network. Add variables to further customize. Separates input
        data into training in test sets. Full dataset should be passed to the network.

        """
        train_idx = random.sample(range(self.num_instances), int(train_size*(self.num_instances)))
        test_idx = [s for s in list(range(self.num_instances)) if s not in train_idx]
        self.Train = (self.Source[train_idx, :, :], self.Target[train_idx])
        self.Test = (self.Source[test_idx, :, :], self.Target[test_idx])

        self.Network = Sequential()
        # return_sequences ensures one state flows into the next:
        self.Network.add(LSTM(32, return_sequences=True,input_shape=(self.timesteps, self.dimensionality)))
        # LSTMs are prone to overfit, so add dropout for regularization:
        self.Network.add(Dropout(0.30))
        self.Network.add(Flatten())
        self.Network.add(Dense(2, activation='softmax'))
        self.Network.compile(loss=objective, optimizer=optimizer, metrics=['accuracy'])

    def Fit(self, epochs=5):
        """
        Fit the model to the training data. Code will report both loss
        and accuracy per epoch during training in the form of a list (e.g.
        >> myModel.Fit()
        ... [loss = 1.55, acc = 0.89]

        """
        self.Network.fit(self.Train[0], self.Train[1], batch_size=self.batch_size, epochs=epochs)
        
    def Evaluate(self):
        """
        Observe network performance on test data.    

        """
        performance = self.Network.evaluate(self.Test[0], self.Test[1], verbose=0)
        return("Accuracy = {0}".format(performance[1]))
    
    def Predict(self, novelData, batch_size):
        """
        Predict classification for new (unlabeled) data after training has succeeded.

        """
        return(self.Network.predict(novelData, batch_size=batch_size, verbose=0))

