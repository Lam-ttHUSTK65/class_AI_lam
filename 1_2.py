import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

# Load the data
filePath = "polydata.dat"
data = np.loadtxt(filePath, delimiter=',')
X = data[:, 0]
y = data[:, 1]

# Polynomial Regression Class
class PolynomialRegression:
    def __init__(self, degree=1, regLambda=1E-8):
        self.degree = degree
        self.regLambda = regLambda
        self.poly = PolynomialFeatures(degree)
        self.model = LinearRegression()

    def fit(self, X, y):
        X_poly = self.poly.fit_transform(X.reshape(-1, 1))
        self.model.fit(X_poly, y)

    def predict(self, X):
        X_poly = self.poly.transform(X.reshape(-1, 1))
        return self.model.predict(X_poly)

# Learning Curve Function
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
        errorTrain[i] = np.mean(err_train ** 2)

        predictTest = model.predict(Xtest)
        err_test = predictTest - Ytest
        errorTest[i] = np.mean(err_test ** 2)

    return (errorTrain, errorTest)

# Generate Learning Curves
def generateLearningCurve(X, y, degree, regLambda):
    n = len(X)
    errorTrains = np.zeros((n-1, n-1))
    errorTests = np.zeros((n-1))

    for i in range(1, n):
        X_train = X[:i]
        y_train = y[:i]

        (errTrain, errTest) = learningCurve(X_train, y_train, X, y, regLambda, degree)

        errorTrains[i-1, :i] = errTrain
        errorTests[i-1] = np.mean(errTest)

    plotLearningCurve(errorTrains.mean(axis=1), errorTests, regLambda, degree)

# Plotting Function
def plotLearningCurve(errorTrain, errorTest, regLambda, degree):
    minX = 3
    maxY = max(errorTest[minX+1:])

    xs = np.arange(len(errorTrain))
    plt.plot(xs, errorTrain, 'r-o', label='Training Error')
    plt.plot(xs, errorTest, 'b-o', label='Testing Error')
    plt.plot(xs, np.ones(len(xs)), 'k--', label='Reference Line')
    plt.legend(loc='best')
    plt.title(f'Learning Curve (d={degree}, lambda={regLambda})')
    plt.xlabel('Training samples')
    plt.ylabel('Error')
    plt.yscale('log')
    plt.ylim((0, maxY))
    plt.xlim((minX, 10))
    plt.show()

# Test different parameters
generateLearningCurve(X, y, degree=1, regLambda=0)
generateLearningCurve(X, y, degree=4, regLambda=0)
generateLearningCurve(X, y, degree=8, regLambda=0)
generateLearningCurve(X, y, degree=8, regLambda=0.1)
generateLearningCurve(X, y, degree=8, regLambda=1)
generateLearningCurve(X, y, degree=8, regLambda=100)
