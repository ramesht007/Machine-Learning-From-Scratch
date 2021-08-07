import numpy as np


class logisticRgerssion:
    """Logistic regression is used to give output as 
        discrete set of classes

    Input Parameters:
    ----------------------------

    """
    def __init__(self, lr = 0.001, n_iters = 1000):
        self.lr = lr
        self.n_iters = n_iters
        # No weights and bias are present at start
        # therefore, weights and bias are None
        self.weights = None
        self.bias = None


    def fit(self, X, y):
        n_samples, n_features = X.shape
        # Initialize weights with respect to the 
        # to the featrues and intialize bias as 0.
        self.weights = np.zeros(n_features)
        self.bias = 0

        # perform gradient for n_iterations
        for _ in range(self.n_iters):
            linear_mdoel = np.dot(X, self.weights) + self.bias
            y_predicted  = self._sigmoid(linear_mdoel)

            # calculate the derivatives of weights and bias 
            # using chain rule
            dw = (1/n_samples) * np.dot(X.T, (y_predicted - y))
            db = (1/n_samples) * np.sum(y_predicted - y)

            # update the weights and bias 
            self.weights -= self.lr * dw
            self.bias -= self.lr * db

    def predict(self, X):
        y_prediction = np.dot(X, self.weights) + self.bias
        prediction = self._sigmoid(y_prediction)
        prediction_cls = [1 if pred > 0.5 else 0 for pred in prediction]
        return prediction_cls

    def _sigmoid(self, X):
        return 1 / (1 + np.exp(-X))
