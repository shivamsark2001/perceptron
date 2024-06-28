import numpy as np

class LinearRegression:
    def __init__(self, lr=0.05, n_iterations=1000):
        self.lr = lr
        self.n_iterations = n_iterations

    def fit(self, X, y):   
        # Initialize weights as a zero array with the same number of columns as X
        self.weights = np.zeros(X.shape[1])
        self.bias = 0

        for _ in range(self.n_iterations):
            y_pred = np.dot(X, self.weights) + self.bias

            # Calculate gradients
            dw = (1 / len(X)) * np.dot(X.T, (y_pred - y))
            db = (1 / len(X)) * np.sum(y_pred - y)

            # Update weights and bias
            self.weights -= self.lr * dw
            self.bias -= self.lr * db

    def predict(self, X):
        return np.dot(X, self.weights) + self.bias
    
    def _mean_squared_error(self, y_true, y_pred):
        return (1 / len(y_true)) * np.sum((y_true - y_pred) ** 2)

from sklearn.datasets import make_regression
import matplotlib.pyplot as plt

# Generate dataset
X, y = make_regression(n_samples=100, n_features=1, noise=10, random_state=42)
print(X.shape) 
print(y.shape) 

# Fit the model
regression = LinearRegression()
regression.fit(X, y)
predictions = regression.predict(X)
mse = regression._mean_squared_error(y, predictions)
print("Mean Squared Error:", mse)

# Plot the results
plt.scatter(X, y)
plt.plot(X, predictions, color='red')
plt.show()
