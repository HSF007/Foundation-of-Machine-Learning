import os
import numpy as np


# Getting file directory to give file path to read file.
directory_name = os.path.dirname(__file__)
train_data_path = os.path.join(directory_name, 'FMLA1Q1Data_train.csv')
test_data_path = os.path.join(directory_name, 'FMLA1Q1Data_test.csv')


# Loading Training data and testing data from given files.
train_data = np.genfromtxt(train_data_path, delimiter=',')
test_data = np.genfromtxt(test_data_path, delimiter=',')


# Adding bias to training and test data. Then taking first two columns as 
# input features and 3rd column from dataset as target column.
train_intercept, test_intercept = np.ones((1000, 1)), np.ones((100, 1))

x_train, y_train = np.append(train_data[:, :2], train_intercept, axis=1), train_data[:, 2]

x_test, y_test = np.append(test_data[:, :2], test_intercept, axis=1), test_data[:, 2]