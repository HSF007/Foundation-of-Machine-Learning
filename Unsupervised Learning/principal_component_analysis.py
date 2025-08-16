import numpy as np

class PCA:
    def __init__(self, components=2):
        self.components = components
    
    def fit(self, X):
        self.mu = np.mean(X, axis=0)
        mean_centered_data = X - self.mu
        n, d = X.shape
        covariance_matrix = 1/n*np.dot(mean_centered_data.T, mean_centered_data)
        self.eigenValues, self.eigenVectors = np.linalg.eigh(covariance_matrix)

        sorted_indices = np.argsort(self.eigenValues)[::-1]
        self.eigenValues = self.eigenValues[sorted_indices]
        self.eigenVectors = self.eigenVectors[:, sorted_indices]

        
        self.seleted_components = self.eigenVectors[:, :self.components]
        self.transformed_data = np.dot(mean_centered_data, self.seleted_components)
    
    def reconstruct(self):
        return np.dot(self.transformed_data, self.seleted_components.T) + self.mu
    
    def get_mean(self):
        return self.mu

    def get_eigenvalues(self):
        return self.eigenValues
    
    def get_eigenvectors(self):
        return self.eigenVectors