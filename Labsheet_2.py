# Import necessary libraries
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
# Load the dataset
file_path = 'Data_LR.csv'  # Replace with your file path
data = pd.read_csv(file_path)
# Preprocess the data
data_cleaned = data.dropna()

# Select features for the model (excluding 'median_house_value' for now)
features = ['longitude', 'latitude', 'housing_median_age', 'total_rooms', 
            'total_bedrooms', 'population', 'households', 'median_income']
X = data_cleaned[features].values
# Target variable
Y = data_cleaned['median_house_value'].values

# Normalize the features (for better gradient descent performance)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Add intercept term (column of 1s) for bias in the linear regression
X_scaled = np.c_[np.ones(X_scaled.shape[0]), X_scaled]
# Split the data into training and test sets
x_train, x_test, y_train, y_test = train_test_split(X_scaled, Y, test_size=0.2, random_state=42)

# Initialize parameters (theta)
theta = np.array([1.5] * X_scaled.shape[1])  # Initial theta values
# Define the cost function
def compute_cost(X, Y, theta):
    m = len(Y)
    predictions = X.dot(theta)
    cost = (1/(2*m)) * np.sum((predictions - Y) ** 2)
    return cost
# Define the gradient descent algorithm
def gradient_descent(X, Y, theta, alpha, iters):
    m = len(Y)
    cost_history = np.zeros(iters)
    
    for i in range(iters):
        predictions = X.dot(theta)
        error = predictions - Y
        gradient = X.T.dot(error) / m
        theta = theta - alpha * gradient
        cost_history[i] = compute_cost(X, Y, theta)
        
        # Adjust learning rate if cost increases
        if i > 0 and cost_history[i] > cost_history[i - 1]:
            alpha *= 0.1
    return theta, cost_history
# Run gradient descent
alpha = 0.05
iterations = 10000
theta_optimized, cost_history = gradient_descent(x_train, y_train, theta, alpha, iterations)

# Plot the cost history over iterations
plt.figure(figsize=(10, 6))
plt.plot(cost_history, color='blue')
plt.xlabel('Iterations')
plt.ylabel('Cost')
plt.title('Cost Function over Iterations')
plt.show()

# Compute predictions for the training data
y_train_pred = x_train.dot(theta_optimized)
# Plot the training data (actual vs predicted)
plt.figure(figsize=(10, 6))
plt.scatter(y_train, y_train_pred, color='red', label='Actual vs Predicted')
plt.plot([min(y_train), max(y_train)], [min(y_train), max(y_train)], color='yellow', label='Perfect Fit Line')
plt.xlabel('Actual Median House Value')
plt.ylabel('Predicted Median House Value')
plt.title('Training Data: Actual vs Predicted Median House Values')
plt.legend()
plt.show()
# Compute predictions for the test data
y_test_pred = x_test.dot(theta_optimized)

# Plot the test data (actual vs predicted)
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_test_pred, color='red', label='Actual vs Predicted')
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='yellow', label='Perfect Fit Line')
plt.xlabel('Actual Median House Value')
plt.ylabel('Predicted Median House Value')
plt.title('Test Data: Actual vs Predicted Median House Values')
plt.legend()
plt.show()
# Calculate Mean Squared Error (MSE)
mse_train = np.mean((y_train_pred - y_train) ** 2)
mse_test = np.mean((y_test_pred - y_test) ** 2)

# Print training and testing errors
print(f"Training MSE: {mse_train}")
print(f"Testing MSE: {mse_test}")