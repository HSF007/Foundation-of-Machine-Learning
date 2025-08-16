import numpy as np


class LogisticRegression:
    def __init__(self, learning_rate=0.01, num_iterations=1000):
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.weights = None
    
    def _add_bias(self, X):
        # Add bias column of ones to given data
        bias = np.ones((X.shape[0], 1))
        return np.hstack([bias, X])
    
    # Sigmoid function
    def _sigmoid(self, z):
        return 1 / (1 + np.exp(-z))
    
    def fit(self, X, y):
        # Add bias to input data
        X = self._add_bias(X)
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        
        # Gradient descent
        for i in range(self.num_iterations):
            # Compute the linear model
            linear_model = np.dot(X, self.weights)
            # Apply sigmoid function
            y_pred = self._sigmoid(linear_model)
            
            # Calculate gradients
            gradient = np.dot(X.T, (y_pred - y)) / n_samples
            # Update weights
            self.weights -= self.learning_rate * gradient
    
    def predict_proba(self, X):
        # Add bias to data for predictions
        X = self._add_bias(X)
        # Calculate  probabilities
        linear_model = np.dot(X, self.weights)
        return self._sigmoid(linear_model)
    
    def predict(self, X):
        # Predict based on a 0.5 threshold
        y_pred_proba = self.predict_proba(X)
        return [1 if i > 0.5 else 0 for i in y_pred_proba]