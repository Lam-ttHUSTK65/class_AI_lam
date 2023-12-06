import numpy as np
import matplotlib.pyplot as plt

class PolynomialRegression:
    def __init__(self, degree=1, regLambda=1E-8):
        self.degree = degree
        self.regLambda = regLambda
        self.theta = None

    def polyfeatures(self, X, degree):
        # Polynomial expansion of X
        X_poly = np.column_stack([X ** i for i in range(1, degree + 1)])
        return X_poly

    def fit(self, X, y):
        # Train the model
        X_poly = self.polyfeatures(X, self.degree)
        X_augmented = np.column_stack([np.ones(X.shape[0]), X_poly])

        identity_matrix = np.identity(X_augmented.shape[1])
        regularization_term = self.regLambda * identity_matrix

        # Closed-form solution for linear regression with regularization
        self.theta = np.linalg.inv(X_augmented.T @ X_augmented + regularization_term) @ X_augmented.T @ y

    def predict(self, X):
        # Make predictions using the trained model
        X_poly = self.polyfeatures(X, self.degree)
        X_augmented = np.column_stack([np.ones(X.shape[0]), X_poly])
        predictions = X_augmented @ self.theta
        return predictions

def learningCurve(Xtrain, Ytrain, Xtest, Ytest, regLambda, degree):
    n = len(Xtrain)
    errorTrain = np.zeros((n))
    errorTest = np.zeros((n))
    
    for i in range(2, n):
        Xtrain_subset = Xtrain[:i + 1]
        Ytrain_subset = Ytrain[:i + 1]
        
        model = PolynomialRegression(degree, regLambda)
        model.fit(Xtrain_subset, Ytrain_subset)
        
        predictTrain = model.predict(Xtrain_subset)
        err_train = predictTrain - Ytrain_subset
        errorTrain[i] = np.multiply(err_train, err_train).mean()

        predictTest = model.predict(Xtest)
        err_test = predictTest - Ytest
        errorTest[i] = np.multiply(err_test, err_test).mean()

    return (errorTrain, errorTest)

# Test script
if __name__ == "__main__":
    filePath = "polydata.dat"
    allData = np.loadtxt(filePath, delimiter=',')

    X = allData[:, 0].reshape(-1, 1)  # Reshape X to a column vector
    y = allData[:, 1].reshape(-1, 1)  # Reshape y to a column vector

    # Regression with degree = d
    d = 8
    model = PolynomialRegression(degree=d, regLambda=0)
    model.fit(X, y)

    # Output predictions
    xpoints = np.linspace(np.max(X), np.min(X), 100).reshape(-1, 1)
    ypoints = model.predict(xpoints)

    # Plot curve
    plt.figure()
    plt.plot(X, y, 'rx')
    plt.title('PolyRegression with d = ' + str(d))
    # plt.hold(True)  # Remove this line; it's not needed
    plt.plot(xpoints, ypoints, 'b-')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.show()
