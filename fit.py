# Import dependencies
import numpy as np
from scipy.odr import *
from sklearn import linear_model
import matplotlib.pyplot as plt

### ======================
### Fitting auxiliary weights
### ======================

def linearFit(X, y):
    """
    Takes a matrix of input data and a vector of output data.
    Tries to map the input data linearly to the output data by
    linear regression.

    Returns vector of weights.
    """
    # If there is no input data, there is only one weight: a constant that
    # is the mean of the output (target) data
    if len(X) == 0:
        return [np.mean(y)]

    lr = linear_model.LinearRegression()
    lr.fit(X.T, y)

    # Run the regression.
    return np.hstack([lr.intercept_, lr.coef_])

def Auxiliaries(X_data, W_sym):
    """
    Identifies the weights that need to be fitted and the corresponding data.
    Forewards this to a fitting function.

    Input: dataset and symbolic weight Matrix.
    Output: a vector of auxiliary weight_values
    """
    w2_num = []
    for n, a in enumerate(np.array(W_sym[:,4:]).T):
        input_data = X_data[:,[bool(w) for w in a[1:]]].T
        output_data = X_data.T[n + 3]
        w2_num += list(linearFit(input_data, output_data))

    return np.array(w2_num)

def stockFit(input_data, output_data, getChange, w_sym):
    """
    Takes a matrix of input data and a vector of output data.
    Tries to map the input data linearly to the output data by
    linear regression.

    Returns vector of weights.
    """
    # Since we cannot estimate the starting weights for the naive models,
    # all initial weights b are set to 0
    b = [0 for n in w_sym]
    real_data = Data(input_data, output_data)

    def f(p, q):
        return getChange(*p, *q)

    odr = ODR(real_data, Model(f), beta0=b)

    # Run the regression.
    return odr.run().beta

def Stress(X_data, W1_data, w3_sym, getNewStress, alpha):
    """
    Does a linear regression equating the current stress value to the stress
    value one timestep later. Effectively minimizing the first order taylor
    approximation

    Returns the weights leading to stress
    """
    last_w3 = np.array([alpha for x in X_data])

    input_data = np.vstack((last_w3.T, X_data.T, W1_data.T))
    output_data = X_data.T[1]
    w3_num = stockFit(input_data, output_data, getNewStress, w3_sym[:-1])

    return np.hstack((w3_num, alpha))

def Weight(X_data, W1_data, w4_sym, getNewWeight):
    """
    Does a linear regression equating the current stress value to the stress 
    value one timestep later. Effectively minimizing the first order taylor
    approximation

    Returns the weights leading to weight
    """
    input_data = np.hstack((X_data, W1_data)).T
    output_data = X_data.T[2]
    w4_num = stockFit(input_data, output_data, getNewWeight, w4_sym)

    return np.array(w4_num)
