import numpy as np
import math

class LinearRegression:
    """Model for Linear problems.

    Input Parameters:
    -----------------------------
    lr: {float}
        It is length of the step used for updating weights 
        while gradient descent.
    n_iters: {int}
        Number of interation for which the model will train 
        the weights. 
    """
    def __init__(self, lr= 0.001, n_iters=1000):
        self.lr = lr
        self.n_iters = n_iters
        # Intially weights and bias are not present. 
        # Therefore, eights and bias are set to None. 
        self.weights = None 
        self.bias = None

    # def _initialize_weights(self, n_features):
    #     """Intialize random weight in range [-1/N, 1/N]
    #     where N is n_features.
    #     """
    #     limit = 1 / math.sqrt(n_features)
    #     self.weights = np.random.uniform(-limit, limit, (n_features, ))

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        # Perform gradient descent for n_iterations
        for _ in range(self.n_iters):
             # Initial approximtion
             # y' = w.x + b
             y_predict = np.dot(X, self.weights) + self.bias

             # calculate derivative for weights and bias
             dw = (1/n_samples) * np.dot(X.T, (y_predict - y))
             db = (1/n_samples) * np.sum(y_predict - y)

             self.weights -= self.lr * dw
             self.bias -= self.lr * db 

    def predict(self, X):
        # approximation of y' = w.x + b
        y_predict = np.dot(X, self.weights) + self.bias
        return y_predict