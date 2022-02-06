#!/usr/bin/env python3
"""
main function
"""
from data_processing import process
from neuron import Neuron
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(0)
X_train, Y_train = process('Binary_Train.npz')
n = Neuron(X_train, Y_train, iteration=5000, alpha = 0.05)
# Intiate a random weight
#n.b = 1

n.train()