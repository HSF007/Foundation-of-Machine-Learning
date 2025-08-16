import numpy as np
import os
import Analytical_solution
import main

# Then taking first two columns as input features and 3rd column
#  from dataset as target column without adding bias as it will be considered in kernel itself.
x_train_kernel, y_train_kernel = main.train_data[:, :2], main.train_data[:, 2]

x_test_kernel, y_test_kernel = main.test_data[:, :2], main.test_data[:, 2]


# Algorithm for Polynomial Kernel regression
class KernelRegression:
    def __init__(self, variance):
        self.variance = variance

    def gaussian_kernel_function(self, x1, x2):
        return np.exp(-np.dot(x1 - x2, x1 - x2)/(2*(self.variance**2)))

    def kernel_matrix(self, X, y):
        data_points, features = X.shape
        kernel = np.zeros((data_points, data_points))
        for i in range(data_points):
            for j in range(data_points):
                kernel[i][j] = self.gaussian_kernel_function(X[i], X[j])
        self.alpha = np.linalg.solve(kernel, y)

    def predicts_using_kernle(self, test_data, X):
        test_data_points, m_test = test_data.shape
        data_points, features = X.shape
        kernel_for_prediction = np.zeros(data_points)
        predicted_values = np.zeros(test_data_points)
        
        for j in range(test_data_points):
            for i in range(data_points):
                kernel_for_prediction[i] = self.gaussian_kernel_function(test_data[j], X[i])
            predicted_values[j] = np.matmul(kernel_for_prediction, self.alpha)
        return predicted_values


# Running algorithm of kernle regression
kernel_regression = KernelRegression(variance=9)

# Creating kernle matrix K and getting alpha
kernel_regression.kernel_matrix(x_train_kernel, y_train_kernel)

# Predicting values for test data
kernel_predictions = kernel_regression.predicts_using_kernle(x_test_kernel, x_train_kernel)

# Getting error for test data
kernel_error = Analytical_solution.squared_error(y_test_kernel, kernel_predictions)
print('Error for Kernle Regression using Polynomial kernle is', kernel_error)