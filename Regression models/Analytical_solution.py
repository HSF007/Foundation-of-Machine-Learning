import numpy as np
import main


x_train_without_bias, x_test_without_bias = main.train_data[:, :2], main.test_data[:, :2]


# Function to find weights using analytical solution
def LinearRegression(X, y):
    denominator = np.matmul(np.transpose(X), X)
    numerator = np.matmul(np.transpose(X), y)
    
    denominator_inverse = np.linalg.inv(denominator)
    w = np.matmul(denominator_inverse, numerator)
    return w


def predicted_values(weights, input_data):
    return np.matmul(input_data, weights)


def squared_error(true_values, predicted_values):
    difference = true_values - predicted_values
    return np.dot(difference, difference)


# Now lets find weights from linear regression function
# and calculate error for test data

# First let's calculate error without bias
w_ml_without_bias = LinearRegression(x_train_without_bias, main.y_train)
predict_target_without_bais = predicted_values(w_ml_without_bias, x_test_without_bias)
ml_error_without_bias = squared_error(main.y_test, predict_target_without_bais)


w_ml = LinearRegression(main.x_train, main.y_train)
predicting_target = predicted_values(w_ml, main.x_test)
ml_error = squared_error(main.y_test, predicting_target)
print('Regression using Analytical Soution error on test data without bais', ml_error_without_bias)
print('Regression using Analytical Soution error on test data', ml_error)