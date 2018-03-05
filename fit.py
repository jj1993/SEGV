# Import dependencies
import numpy as np
from scipy.odr import *
from scipy.optimize import basinhopping

### ======================
### Fitting auxiliary weights
### ======================

def f(p, q):
    """
    Multiplies all data points q (matrix) by the weights p (vector),
    where p[0] is multiplied by 1 (a constant added to the linear equation).

    Returns vector of multiplied values.
    """
    if q.ndim == 1:
        # Debug for case where f = p[0] + p[1]*q
        # Scipy tries to change q from shape (1,N) to (N,)
        q = q.reshape((1,len(q)))
    dummy_q = np.vstack(([1 for n in range(len(q[0]))],q))
    return np.dot(dummy_q.T, p)

def linearFit(input_data, output_data):
    """
    Takes a matrix of input data and a vector of output data.
    Tries to map the input data linearly to the output data by
    orthogonal distance regression.

    Returns vector of weights.
    """
    # If there is no input data, there is only one weight: a constant that
    # is the mean of the output (target) data
    if len(input_data) == 0: return [np.mean(output_data)]
    # Since we cannot estimate the starting weights for the naive models,
    # all initial weights b are set to 0
    b = [0 for n in range(len(input_data) + 1)]
    real_data = Data(input_data, output_data)
    odr = ODR(real_data, Model(f), beta0=b)

    # Run the regression.
    return odr.run().beta

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

def log_likelihood(w3_num, X_data, W1_data, w2_num, getL_n):
    L = getL_n(*X_data.T, *W1_data.T, *w2_num, *w3_num)
    return np.mean(L)

# def is_attractor(f_new, x_new, f_old, x_old):
#     L = num_attract(*X_data.T, *W1_data.T, *W2_num, *x_new)
#
#     return np.all([np.mean(l) > 0 for l in L])

def Stocks(X_data, W1_data, w2_num, getL_n, w3_init):
    minimizer_kwargs = {
        "args": (X_data, W1_data, w2_num, getL_n),
        # "method": "Powell"
    }
    res = basinhopping(
                log_likelihood, w3_init,
                minimizer_kwargs = minimizer_kwargs,
                niter = 100,
                stepsize = .1,
                T = 1e-2,
                # disp = True,
                # accept_test = is_attractor
            )
    return res.x
