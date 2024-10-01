import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from numpy.polynomial.polynomial import polyfit, polyval

# CODE IS TAKEN FROM THE PROVIDED BLOG AND PROVIDED CODE THAT WAS PROVIDED IN THE ASSIGNMENT
# https://dustinstansbury.github.io/theclevermachine/bias-variance-tradeoff

# Function f(x) = x + sin(1.5x)
def f(x):
    return x + np.sin(1.5 * x)

# Constants
np.random.seed(124)
n_observations_per_dataset = 50  
n_datasets = 100  
max_poly_degree = 15  
model_poly_degrees = range(1, max_poly_degree + 1)

NOISE_STD = np.sqrt(0.3)  # Noise standard deviation
percent_train = 0.8
n_train = int(np.ceil(n_observations_per_dataset * percent_train))

# Create training/testing inputs
x = np.linspace(-1, 1, n_observations_per_dataset)
x = np.random.permutation(x)  # Shuffle x
x_train = x[:n_train]
x_test = x[n_train:]

# Logging variables
theta_hat = defaultdict(list)
pred_train = defaultdict(list)
pred_test = defaultdict(list)
train_errors = defaultdict(list)
test_errors = defaultdict(list)

def error_function(pred, actual):
    return (pred - actual) ** 2

# Loop over datasets
for dataset in range(n_datasets):
    y_train = f(x_train) + NOISE_STD * np.random.randn(*x_train.shape)
    y_test = f(x_test) + NOISE_STD * np.random.randn(*x_test.shape)

    for degree in model_poly_degrees:
        tmp_theta_hat = polyfit(x_train, y_train, degree)

        # Make predictions on train set
        tmp_pred_train = polyval(x_train, tmp_theta_hat)  # Changed order of arguments
        #tmp_pred_train = polyval(tmp_theta_hat, x_train)  # Correct order

        pred_train[degree].append(tmp_pred_train)

        # Debug: Print shapes before calling error_function
        print(f"Shape of tmp_pred_train: {tmp_pred_train.shape}")
        print(f"Shape of y_train: {y_train.shape}")

        # No need for reshaping now, as shapes should match
        train_errors[degree].append(np.mean(error_function(tmp_pred_train, y_train)))

        # Test predictions
        tmp_pred_test = polyval(x_test, tmp_theta_hat)  # Changed order of arguments
        #tmp_pred_test = polyval(tmp_theta_hat, x_test)
        pred_test[degree].append(tmp_pred_test)

        # Mean Squared Error for train and test sets
        test_errors[degree].append(np.mean(error_function(tmp_pred_test, y_test)))


# Functions to calculate bias^2 and variance
def calculate_estimator_bias_squared(pred_test):
    pred_test = np.array(pred_test)
    average_model_prediction = pred_test.mean(0)  # E[g(x)]

    # (E[g(x)] - f(x))^2, averaged across all trials
    return np.mean((average_model_prediction - f(x_test)) ** 2)


def calculate_estimator_variance(pred_test):
    pred_test = np.array(pred_test)
    average_model_prediction = pred_test.mean(0)  # E[g(x)]

    # (g(x) - E[g(x)])^2, averaged across all trials
    return np.mean((pred_test - average_model_prediction) ** 2)


# Bias-Variance calculations
complexity_train_error = []
complexity_test_error = []
bias_squared = []
variance = []

for degree in model_poly_degrees:
    complexity_train_error.append(np.mean(train_errors[degree]))
    complexity_test_error.append(np.mean(test_errors[degree]))
    bias_squared.append(calculate_estimator_bias_squared(pred_test[degree]))
    variance.append(calculate_estimator_variance(pred_test[degree]))

best_model_degree = model_poly_degrees[np.argmin(complexity_test_error)]


# Visualizations
fig, axs = plt.subplots(1, 2, figsize=(14, 10))

# Plot Bias^2 + Variance
plt.sca(axs[0])
plt.plot(model_poly_degrees, bias_squared, color='blue', label='$bias^2$')
plt.plot(model_poly_degrees, variance, color='green', label='variance')
plt.plot(model_poly_degrees, np.array(bias_squared) + np.array(variance), linestyle='-.', color='gray', label='$bias^2 + variance$')
plt.plot(model_poly_degrees, complexity_test_error, label='Testing Set Error', linewidth=3, color='red')
plt.axvline(best_model_degree, linestyle='--', color='black', label=f'Best Model (degree={best_model_degree})')

plt.xlabel('Model Complexity (Polynomial Degree)')
plt.ylim([0, 1])
plt.legend()
plt.title('Testing Error vs Bias and Variance')

# Plot Train / Test Set Error
plt.sca(axs[1])
plt.plot(model_poly_degrees, complexity_train_error, label='Training Set Error', linewidth=3, color='blue')
plt.plot(model_poly_degrees, complexity_test_error, label='Testing Set Error', linewidth=3, color='red')
plt.axvline(best_model_degree, linestyle='--', color='black', label=f'Best Model (degree={best_model_degree})')

plt.ylim([0, 1])
plt.xlabel('Model Complexity (Polynomial Degree)')
plt.title('Error on Training and Testing Sets')
plt.legend(loc='upper center')

plt.show()
