import numpy as np

class Llyoid_Algorithm:
    def __init__(self, number_of_clustors=2):
        self.number_of_clustors = number_of_clustors

    def fit(self, X, iterations=100, random_state=400):
        np.random.seed(random_state)
        self.error = []
        n, d = X.shape
        data_labels = np.random.randint(0, self.number_of_clustors, size=n)
        for _ in range(iterations):
            means = np.array([(X[data_labels == i]).mean(axis=0) for i in range(self.number_of_clustors)])
            distances = np.linalg.norm(X[:, np.newaxis] - means, axis=2)
            data_labels = np.argmin(distances, axis=1)
            self.error.append(np.sum((X - means[data_labels])**2)/n)
        self.means = np.array([(X[data_labels == i]).mean(axis=0) for i in range(self.number_of_clustors)])
        return data_labels
    
    def get_error(self):
        return self.error
    
    def get_mean(self):
        return self.means