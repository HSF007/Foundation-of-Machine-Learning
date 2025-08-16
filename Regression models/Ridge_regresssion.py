import numpy as np
import matplotlib.pyplot as plt
import main
import Analytical_solution


# Code for K-Cross Validation
def KcrossValidation(num_data, k):
    np.random.seed(21)
    indices = np.arange(num_data)
    np.random.shuffle(indices)
    batch = int(num_data/k)
    validation_indices = indices.reshape(k, batch)
    train_i = np.array(np.expand_dims(indices[batch:], axis=0))
    for i in range(1, k):
        start = batch*i
        end = start + batch
        train_i = np.append(train_i, np.expand_dims(np.concatenate((indices[:start], indices[end:])
                                                                   , axis=0), axis=0), axis=0)
    return train_i, validation_indices


# Algorithm for Ridge Regression
def RidgeRegression(x, y, iterations, step_size, lambda_val):
    n, features = x.shape
    weights = np.zeros(features)
    xTy = np.matmul(np.transpose(x), y)
    xTx = np.matmul(np.transpose(x), x)
    for _ in range(iterations):
        gradient_of_w = 2*np.matmul(xTx, weights) - 2*xTy + 2*lambda_val*weights
        weights = weights - step_size*(gradient_of_w)
    return weights


def choosing_lambda(iterations, step_size, lambda_values, k, number_of_data):
    validation_errors = []
    k_train_index, k_test_index = KcrossValidation(number_of_data, k)

    for val in lambda_values:
        validate_error = 0
        for train_i, validate_i in zip(k_train_index, k_test_index):
            w_ridge= RidgeRegression(main.x_train[train_i],
                                     main.y_train[train_i],
                                     iterations, step_size, val)

            y_predict = Analytical_solution.predicted_values(w_ridge, main.x_train[validate_i])

            validate_error += Analytical_solution.squared_error(main.y_train[validate_i], y_predict)
        validation_errors.append(validate_error/k)
    return validation_errors

# Creating shuffled Indecies for K-cross
number_of_data, features = main.x_train.shape
iterations, k = 1500, 5
step_size = 0.0001

temp_lambda1 = np.arange(1, 100)
temp_lambda2 = np.arange(1, 10)
temp_lambda3 = np.arange(2.5, 3.5, 0.01)

valid_errors1 = choosing_lambda(iterations, step_size, temp_lambda1, k, number_of_data)

valid_errors2 = choosing_lambda(iterations, step_size, temp_lambda2, k, number_of_data)

valid_errors3 = choosing_lambda(iterations, step_size, temp_lambda3, k, number_of_data)


# Running Algorithm
plt.plot(temp_lambda1, valid_errors1)
plt.xlabel('$\\lambda$')
plt.ylabel('Validation Error')
plt.title('Loss Plot')
plt.show()


plt.plot(temp_lambda2, valid_errors2)
plt.xlabel('$\\lambda$')
plt.ylabel('Validation Error')
plt.title('Loss Plot')
plt.show()


plt.plot(temp_lambda3, valid_errors3)
plt.xlabel('$\\lambda$')
plt.ylabel('Validation Error')
plt.title('Loss Plot')
plt.show()


# Finding lambda which gives least error
min_val = float('inf')
index = None
for i in range(len(valid_errors3)):
    if valid_errors3[i] < min_val:
        min_val = valid_errors3[i]
        index = i
print('Best value for lambda is', temp_lambda3[index])

lambda_val = temp_lambda3[index]
# So we got best lambda = 4.46. Let's obtain w_R
w_r = RidgeRegression(main.x_train, main.y_train,
                      iterations=2000, step_size=0.0001, lambda_val=lambda_val)


# Now let's compare error of w_R with w_ML
ridge_predictions = Analytical_solution.predicted_values(w_r, main.x_test)

ridge_error = Analytical_solution.squared_error(main.y_test, ridge_predictions)

print('For best lambda w_r will be', w_r)
print('Squared error using Ridge Regression is', ridge_error)
print('Comparing error\'s from w_ML and w_R', abs(ridge_error-Analytical_solution.ml_error))

# Ridge regression gave less error and ridge regression is better because it keeps weights 
# of features under check by penalizing them. More the weights for features are then the grater regulization term.
# Which is lambda time squared of it's features weight. Which helps in decreasing biase.