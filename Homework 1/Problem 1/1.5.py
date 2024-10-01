from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression, Ridge
import numpy as np
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_squared_error

degree = 10
n_datasets = 100
n_train = 40
n_test = 10
mu = 0
sigma = np.sqrt(0.3)

def f(x):
    return x + np.sin(1.5 * x)


# Constants
START = -1
END = 1
TOTAL_SAMPLES = 50
NOISE_MEAN = mu  # Assuming mu is defined elsewhere
NOISE_STD = sigma  # Assuming sigma is defined elsewhere
POLYNOMIAL_DEGREE = degree  # Assuming degree is defined elsewhere

def generate_noisy_data(x_data, true_function):
    return true_function(x_data) + np.random.normal(NOISE_MEAN, NOISE_STD, x_data.shape)

def predict_and_evaluate(model, feature_transformer, x_data, y_true):
    x_transformed = feature_transformer.transform(x_data.reshape(-1, 1))
    y_pred = model.predict(x_transformed)
    mse = mean_squared_error(y_true, y_pred)
    return y_pred, mse

# Initialize result lists
standard_predictions = []
regularized_predictions = []
standard_mse_list = []
regularized_mse_list = []

# Generate and shuffle x values
x = np.linspace(start=START, stop=END, num=TOTAL_SAMPLES)
x = np.random.permutation(x)

# Split data into train and test sets
x_train = x[:n_train]
x_test = x[n_train:]

# Generate noisy test data
y_test = f(x_test) + np.random.normal(loc=NOISE_MEAN, scale=NOISE_STD, size=x_test.shape)


# Fit polynomial regression models
def fit_polynomial_model(features, targets, polynomial_degree, use_regularization=False):
    # Create polynomial features
    feature_transformer = PolynomialFeatures(degree=polynomial_degree)
    transformed_features = feature_transformer.fit_transform(features.reshape(-1, 1))
    
    # Choose model based on regularization flag
    if use_regularization:
        regression_model = Ridge(alpha=1.0)
    else:
        regression_model = LinearRegression()
    
    # Fit the model
    regression_model.fit(transformed_features, targets)
    
    return regression_model, feature_transformer


# Main loop
for _ in range(n_datasets):
    # Generate noisy training and test data
    y_train_noisy = generate_noisy_data(x_train, f)
    y_test_noisy = generate_noisy_data(x_test, f)

    # Fit and evaluate standard polynomial model
    standard_model, standard_features = fit_polynomial_model(x_train, y_train_noisy, POLYNOMIAL_DEGREE)
    standard_pred, standard_mse = predict_and_evaluate(standard_model, standard_features, x_test, y_test_noisy)
    standard_predictions.append(standard_pred)
    standard_mse_list.append(standard_mse)

    # Fit and evaluate regularized polynomial model
    regularized_model, regularized_features = fit_polynomial_model(x_train, y_train_noisy, POLYNOMIAL_DEGREE, use_regularization=True)
    regularized_pred, regularized_mse = predict_and_evaluate(regularized_model, regularized_features, x_test, y_test_noisy)
    regularized_predictions.append(regularized_pred)
    regularized_mse_list.append(regularized_mse)

# Convert list of test predictions to numpy array for easier calculation
standard_predictions = np.array(standard_predictions)
regularized_predictions = np.array(regularized_predictions)


# Calculate average prediction across all datasets and model types
def calculate_model_performance_metrics(all_predictions, error_scores, true_x_values, true_function):
    # Calculate average prediction across all datasets
    mean_prediction = np.mean(all_predictions, axis=0)
    
    # Calculate bias squared
    true_y_values = true_function(true_x_values)
    bias_squared = np.mean((true_y_values - mean_prediction) ** 2)
    
    # Calculate variance
    prediction_variance = np.mean(np.var(all_predictions, axis=0))
    
    # Calculate average mean squared error
    average_mse = np.mean(error_scores)
    
    return mean_prediction, bias_squared, prediction_variance, average_mse


def calculate_and_unpack_metrics(predictions, mse_list, x_values, true_function):
    return calculate_model_performance_metrics(predictions, mse_list, x_values, true_function)

# Model types
STANDARD = "Standard"
REGULARIZED = "Regularized"

# Calculate and unpack metrics for both models
models = {
    STANDARD: (standard_predictions, standard_mse_list),
    REGULARIZED: (regularized_predictions, regularized_mse_list)
}

results = {}
for model_type, (predictions, mse_list) in models.items():
    mean_prediction, bias_squared, variance, avg_mse = calculate_and_unpack_metrics(
        predictions, mse_list, x_test, f
    )
    results[model_type] = {
        "mean_prediction": mean_prediction,
        "bias_squared": bias_squared,
        "variance": variance,
        "avg_mse": avg_mse
    }

# Access results
avg_prediction_test, bias_squared_g10, variance_g10, avg_mse_g10 = results[STANDARD].values()
avg_prediction_test_g10_reg, bias_squared_g10_reg, variance_g10_reg, avg_mse_g10_reg = results[REGULARIZED].values()

print(f"Standard Polynomial Model (Degree 10):")
print(f"  Average Mean Squared Error: {avg_mse_g10:.4f}")
print(f"  Prediction Variance: {variance_g10:.4f}")
print(f"  Squared Bias: {bias_squared_g10:.4f}\n")

print(f"Regularized Polynomial Model (Degree 10, L2 Regularization):")
print(f"  Average Mean Squared Error: {avg_mse_g10_reg:.4f}")
print(f"  Prediction Variance: {variance_g10_reg:.4f}")
print(f"  Squared Bias: {bias_squared_g10_reg:.4f}")
