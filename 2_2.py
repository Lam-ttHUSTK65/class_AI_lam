import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class LogisticRegression:
    def __init__(self, alpha=0.01, reg_lambda=0.01, epsilon=1e-4, max_num_iters=10000):
        '''
        Constructor
        '''
        self.alpha = alpha
        self.reg_lambda = reg_lambda
        self.epsilon = epsilon
        self.max_num_iters = max_num_iters
        self.theta = None

    def sigmoid(self, Z):
        '''
        Computes the sigmoid function 1/(1+exp(-z))
        '''
        return 1 / (1 + np.exp(-Z))

    def compute_cost(self, theta, X, y, reg_lambda):
        '''
        Computes the objective function
        Arguments:
            X is a n-by-d numpy matrix
            y is an n-dimensional numpy vector
            reg_lambda is the scalar regularization constant
        Returns:
            a scalar value of the cost
        '''
        m = len(y)
        h = self.sigmoid(np.dot(X, theta))
        reg_term = (reg_lambda / (2 * m)) * np.sum(theta[1:]**2)
        J = (-1 / m) * (np.dot(y.T, np.log(h)) + np.dot((1 - y).T, np.log(1 - h))) + reg_term
        return J[0, 0]  # Return a scalar, not a 1x1 matrix

    def compute_gradient(self, theta, X, y, reg_lambda):
        '''
        Computes the gradient of the objective function
        Arguments:
            X is a n-by-d numpy matrix
            y is an n-dimensional numpy vector
            reg_lambda is the scalar regularization constant
        Returns:
            the gradient, a d-dimensional vector
        '''
        m = len(y)
        h = self.sigmoid(np.dot(X, theta))
        reg_term = (reg_lambda / m) * np.concatenate(([0], theta[1:]))
        gradient = (1 / m) * np.dot(X.T, (h - y)) + reg_term.reshape((-1, 1))
        return gradient

    def fit(self, X, y):
        '''
        Trains the model
        Arguments:
            X is a n-by-d numpy matrix
            y is an n-dimensional numpy vector
        '''
        n, d = X.shape
        X = np.c_[np.ones((n, 1)), X]  # Augment with a column of ones for theta0

        # Initialize theta with random values
        self.theta = np.random.rand(d + 1, 1)

        prev_theta = np.copy(self.theta)
        for i in range(self.max_num_iters):
            gradient = self.compute_gradient(self.theta, X, y, self.reg_lambda)
            self.theta = self.theta - self.alpha * gradient

            # Check convergence
            if np.linalg.norm(self.theta - prev_theta, ord=2) < self.epsilon:
                break

            prev_theta = np.copy(self.theta)

    def predict(self, X):
        '''
        Used the model to predict values for each instance in X
        Arguments:
            X is a n-by-d numpy matrix
        Returns:
            an n-dimensional numpy vector of the predictions
        '''
        n, _ = X.shape
        X = np.c_[np.ones((n, 1)), X]  # Augment with a column of ones for theta0
        probabilities = self.sigmoid(np.dot(X, self.theta))
        predictions = (probabilities >= 0.5).astype(int)
        return predictions

# Load Data
filename = 'data1.dat'
data = pd.read_csv(filename, header=None, names=['Exam1', 'Exam2', 'Admitted'])

# Extract features and labels
X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values

# Convert class labels to 0 and 1
y = np.where(y == 'Admitted', 1, 0)

# Standardize the data
mean = X.mean(axis=0)
std = X.std(axis=0)
X = (X - mean) / std

# Train logistic regression
logreg_model = LogisticRegression(reg_lambda=0.00000001)
logreg_model.fit(X, y)

# Plot the decision boundary
h = .02  # step size in the mesh
x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
Z = logreg_model.predict(np.c_[xx.ravel(), yy.ravel()])

# Put the result into a color plot
Z = Z.reshape(xx.shape)
plt.figure(1, figsize=(4, 3))
plt.pcolormesh(xx, yy, Z, cmap=plt.cm.Paired)

# Plot the training points
plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k', cmap=plt.cm.Paired)

# Configure the plot display
plt.xlabel('Exam 1 Score')
plt.ylabel('Exam 2 Score')
plt.xlim(xx.min(), xx.max())
plt.ylim(yy.min(), yy.max())
plt.xticks(())
plt.yticks(())

plt.show()
