import numpy as np
import math
import sys 


def euclidean_distance(x1, x2):
    """Calculate distance between the two given vectors."""
    return np.sqrt(np.sum((x1-x2)**2))

def get_accuracy(predictions, y_test):
    """Compare two given vectors and return accuracy score."""
    acc = np.sum(predictions == y_test, axis=0) / len(y_test)
    return acc

def mean_squared_error(predictions, y_test):
    """Reutrn mean squared error for y_test, predictions"""
    mse = np.mean((y_test, predictions)**2)
    return mse

