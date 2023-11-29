'''
    TEST SCRIPT FOR MULTIVARIATE LINEAR REGRESSION
    AUTHOR Eric Eaton, Vishnu Purushothaman Sreenivasan
'''

'''
Numpy is a standard library in python that lets you do matrix and vector operations like Matlab in python.
Check out documentation here: http://wiki.scipy.org/Tentative_NumPy_Tutorial
If you are a Matlab user this page is super useful: http://wiki.scipy.org/NumPy_for_Matlab_Users 
'''
import numpy as np
from numpy.linalg import *

# our linear regression class
from linreg import LinearRegression


if __name__ == "__main__":
    '''
    Main function to test univariate and multivariate linear regression
    '''

    # Load the data
    file_path_univariate = "data/univariateData.dat"
    file_path_multivariate = "data/multivariateData.dat"

    univariate_data = np.loadtxt(file_path_univariate, delimiter=',')
    multivariate_data = np.loadtxt(file_path_multivariate, delimiter=',')

    X_univariate = np.matrix(univariate_data[:, :-1])
    y_univariate = np.matrix((univariate_data[:, -1])).T

    X_multivariate = np.matrix(multivariate_data[:, :-1])
    y_multivariate = np.matrix((multivariate_data[:, -1])).T

    n_univariate, d_univariate = X_univariate.shape
    n_multivariate, d_multivariate = X_multivariate.shape

    # Add a row of ones for the bias term
    X_univariate = np.c_[np.ones((n_univariate, 1)), X_univariate]
    X_multivariate = np.c_[np.ones((n_multivariate, 1)), X_multivariate]

    # Initialize the model
    init_theta_univariate = np.matrix(np.ones((d_univariate + 1, 1))) * 10
    init_theta_multivariate = np.matrix(np.ones((d_multivariate + 1, 1))) * 10

    alpha = 0.01
    n_iter = 1500

    # Instantiate objects
    lr_model_univariate = LinearRegression(init_theta=init_theta_univariate, alpha=alpha, n_iter=n_iter)
    lr_model_multivariate = LinearRegression(init_theta=init_theta_multivariate, alpha=alpha, n_iter=n_iter)

    # Train the models
    lr_model_univariate.fit(X_univariate, y_univariate)
    lr_model_multivariate.fit(X_multivariate, y_multivariate)

    # Print the closed-form solutions
    closed_form_solution_univariate = inv(X_univariate.T @ X_univariate) @ X_univariate.T @ y_univariate
    print("Closed-form solution (univariate):", closed_form_solution_univariate)

    closed_form_solution_multivariate = inv(X_multivariate.T @ X_multivariate) @ X_multivariate.T @ y_multivariate
    print("Closed-form solution (multivariate):", closed_form_solution_multivariate)

    # ... (continue with the rest of your code)
