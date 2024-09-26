#Generating 2D data
import math
import numpy as np
from sklearn.datasets import make_classification
x,y = make_classification(n_samples = 100, n_features=2, n_informative=2, n_redundant=0, n_repeated=0, n_classes=2)
def sigmoid(z):
    return 1 / (1+np.exp(-z))
def cost_function(y,y1):
    loss =-np.mean(y*np.log(y1)+(1-y)*np.log(1-y1))
    return loss
 #Compute gradients
def gradients(X, y, y1):
    m = X.shape[0]
    dw = (1 / m) * np.dot(X.T, (y1 - y))
    db = (1 / m) * np.sum(y1 - y)
    return dw, db
# Train the model
def train(X, y, iters, alpha):
    m, n = X.shape
    w = np.zeros((n, 1))
    b = 0
    y = y.reshape(m, 1)
    cost_history = []
    
    for i in range(iters):
        # Forward pass: compute predictions
        z = np.dot(X, w) + b
        y1 = sigmoid(z)

        # Compute gradients
        dw, db = gradients(X, y, y1)

        # Update weights and bias
        w -= alpha * dw
        b -= alpha * db

        # Compute and store the cost
        cost = cost_function(y, y1)
        cost_history.append(cost)

        if i % 100 == 0:
            print(f"Iteration {i}, Cost: {cost}")

        return w, b, cost_history
    # Prediction function
def predict(X, w, b):
    z = np.dot(X, w) + b
    y1 = sigmoid(z)
    return [1 if i > 0.5 else 0 for i in y1]
# Split the data into training and test sets
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
# Training the model
w_new, b_new, cost_history = train(x_train, y_train, iters=1000, alpha=0.001)
# Printing the weights and bias
print("New weights are:", w_new)
print("New intercept is:", b_new)
# Making predictions
predictions = predict(x_test, w_new, b_new)
print("prediction:", predictions)
from sklearn.metrics import accuracy_score, precision_score, recall_score
def evaluate_model(y_true, y_pred):
  #Accuracy
  accuracy = accuracy_score(y_true, y_pred)
  # Precision
  precision = precision_score(y_true, y_pred)
  #Recall
  recall = recall_score(y_true, y_pred)
  return accuracy, precision, recall
accuracy, precision, recall = evaluate_model(y_test, predictions)
print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)