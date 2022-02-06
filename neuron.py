#!/usr/bin/env python3
"""
Neuron class for logistic regression
"""

import numpy as np

class Neuron:
    """
    Single Neuron for logistic regression 
    """

    def __init__(self, X, Y, iteration, alpha):
        """
        X = Input matrix of shape(nx, m) where nx is length of each input vector and m is number of trainig example
        Y = trainig labes
        iterations = number of iterations
        alpha = trainig rate
        """

        self.m = X.shape[1]
        self.nx = X.shape[0]
        self.X = X
        self.Y = Y
        self.alpha = alpha
        self.iteration = iteration
        #Intialize random weights
        self.W = np.random.randn(1, self.nx)
        self.b = 0 
        # The neuron output -> at initialization is equal to zero
        self.A = 0
    
    # Forward propagation pass

    def sigmoid(self, x):
        """
        Sigmoid function
        """
        return 1 / (1 + np.exp(-x))

    def forward_propagation(self):
        """
        Forward propagatrion pass using sigmoid activation function
        """
        a = np.matmul(self.W, self.X)  + self.b
        z = self.sigmoid(a)
        self.A = z
        #print(len(self.X))
        return self.A

    # Defina a cost function for logistic regression
    def cost(self):
        """
        Cost function for logistic regression
        """
        cost =  - np.sum(self.Y * np.log(self.A) + (1 - self.Y) * np.log(1.0000001 - self.A)) / self.m
        return cost

    
    # define theshold
    def evaluate(self):
        """
        define P if A is > 0.5 the 1 else 0
        """
        predections = np.where(self.A < 0.5, 0, 1)
        return predections, self.cost()

    
    # define the backpropagation pass using gradient descent
    def gradient_descent(self):
        """
        Gradient descnet using derivative and chain rule for the cost function in respect to W and B
        """
        dz = self.A - self.Y

        # dw the sum of all derivatives of cost function in respect to Wi
        dw = np.sum(self.X * dz, axis=1)

        #db the derivate of cost function of cost function in respect to b
        db = np.sum(dz)

        # Update the parameters W and b

        self.w = self.W - (self.alpha * (dw / self.m))
        self.b = self.b - (self.alpha * (db /self.m))

    
    def train(self):
        """
        Train the neuron
        """
        
        for i in range(self.iteration + 1):
            self.forward_propagation()
            self.gradient_descent()

            if i == 0 or i == self.iteration + 1 or i % 100 == 0:
                print(f"Cost at iteration {i} = {self.cost()}")
        return self.evaluate()