import numpy as np
import matplotlib.pyplot as plt
import main
import Analytical_solution


# Algorithm for Stochastic Gradient Descent
def stoch_gradient_descent(x, y, iterations, step_size, batch_size, W_ml):
    number_of_data_points, features = x.shape
    weights = np.zeros(features) # Initilaizing weights as zero
    weight_sum = np.zeros(features)
    wt_Wml_norm = []

    for _ in range(iterations):
        random_indecies = np.random.choice(1000, size=batch_size)

        xTy = np.matmul(np.transpose(x[random_indecies]), y[random_indecies])
        xTx = np.matmul(np.transpose(x[random_indecies]), x[random_indecies])

        weights = weights - step_size*(2*np.matmul(xTx, weights) - 2*xTy)

        weight_sum += weights
        wt_Wml_norm.append(np.linalg.norm(weights - W_ml))
    return weight_sum/iterations, wt_Wml_norm


# Running Algorithm
iterations = 100
step_size = 0.001
batch_size=100
stoch_weights, stoch_norm_vals = stoch_gradient_descent(main.x_train, main.y_train,
                                        iterations, step_size, batch_size,
                                        Analytical_solution.w_ml)

# Predicting target values for test data
stoch_predicts = Analytical_solution.predicted_values(stoch_weights, main.x_test)

# Calculating error for test data
stoch_error = Analytical_solution.squared_error(main.y_test, stoch_predicts)

print('Stochastic Gradient Descent error on test data', stoch_error)

# Plotting graph as aked in question
plt.plot(np.arange(iterations), stoch_norm_vals)
plt.title('$||w^t - w_{ML}||_2$ vs t Plot')
plt.xlabel('t-Interations')
plt.ylabel('$||w^t - w_{ML}||_2$')
plt.show()