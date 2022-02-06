#!/usr/bin/env python3
"""
Process MNIST binary dataset
"""
import numpy as np

def process(dataset):
    """
    Process and create array from image
    dataset numpy object
    """
    data = np.load(dataset)
    x_3d, Y = data['X'], data['Y']

    # Flatten images
    X = x_3d.reshape(x_3d.shape[0], x_3d.shape[1] * x_3d.shape[2]).T
    #print(X.shape)
    #print(Y.shape)
    return X, Y


if __name__ == '__main__':
    process('Binary_Train.npz')