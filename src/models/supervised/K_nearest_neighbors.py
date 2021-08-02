import numpy as np
from src.utils.data_operation import euclidean_distance

class KNN():
    """ K Nearest Neighbor classifier.
    --------
    input:
        k : {int}
        Number of nearest neighbors that will determine 
        the class or value of prediction.    
    """
    def __init__(self, k=3):
        self.k = k

    def fit(self, X, y):
        # store the training samples for latter use
        self.X_train = X
        self.y_train = y

    def predict(self, X):
        # this function can get single or multiple samples at a time 
        predicted_labels = [self._predict(x) for x in X]
        return np.array(predicted_labels )

    def _predict(self, x):
        # this method will be passed with one sample at a time 
        # Compute distances
        distances = [euclidean_distance(x, x_train) for x_train in self.X_train]
        # k nearest samples, labels
        k_samples_index = np.argsort(distances)[:self.k] # sort the distances and select top k samples
        k_nearest_label = [self.y_train[i] for i in k_samples_index] # get the labels based on index from k_sample_index 
        # majority vote, most common class model
        most_common = self._vote(np.array(k_nearest_label))
        return most_common         

    def _vote(self, neighbor_labels):
        """ Return the most common class among the neighbor samples """
        counts = np.bincount(neighbor_labels.astype('int'))
        return counts.argmax()