import numpy as np
import matplotlib.pyplot as plt
import main
import Analytical_solution


# Algorith for Gradient Descent
def GradientDescent(X, y, iterations, step_size, W):
    number_of_data_points, features = X.shape

    weights =   np.random.standard_normal(size = features)

    wt_Wml_norm = []

    xTy = np.matmul(np.transpose(X), y)
    xTx = np.matmul(np.transpose(X), X)

    for _ in range(iterations):
        weights = weights - step_size*(2*np.matmul(xTx, weights) - 2*xTy)
        wt_Wml_norm.append(np.linalg.norm(weights - W))
    return weights, wt_Wml_norm


# Now lets find weights from Gradient Descent function
# and calculate error for test data and plot w_t - w_ml norm

iterations = 50
step_size = 0.0001

gradient_weights, norm_values = GradientDescent(main.x_train, main.y_train,
                                                iterations,
                                                step_size,
                                                Analytical_solution.w_ml)


# Predicting target values for test data
gradient_predicts = Analytical_solution.predicted_values(gradient_weights,
                                                                    main.x_test)

# Calculating error for test data
gradient_error = Analytical_solution.squared_error(main.y_test, gradient_predicts)

print('Gradient Descent Error on test data', gradient_error)

# Plotting graph as aked in question
plt.plot(np.arange(iterations), norm_values)
plt.title('$||w^t - w_{ML}||_2$ vs t Plot')
plt.xlabel('t-Interations')
plt.ylabel('$||w^t - w_{ML}||_2$')
plt.show()