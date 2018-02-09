import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import sympy as sp
from sympy.tensor.array import derive_by_array as Derive
from sympy.matrices.dense import hessian as Hessian

def getWeightMatrix(W, Fixed, shuffle=False):
    if shuffle:
        # Creating a valid shuffled graph
        weights = W_expert.T.flatten()[18:]
        np.random.shuffle(weights)
        W = np.array([0 for i in range(18)] + list(weights)).reshape((9,9)).T
        A = [0 if w == 0 else 1 for w in (W+Fixed).flatten()]
        G = nx.Graph(np.array(A).reshape((9,9)))
        # nx.draw(G)
        # plt.show()

        # Checks that no weights overlap the literature weights and
        # that the graph is connected
        while sum(
            [not (type(w) == int or type(f) == int) for w, f in
             zip(W.flatten(), Fixed.flatten())]
        ) != 0 or not nx.is_connected(G):
            np.random.shuffle(weights)
            W = np.array([0 for i in range(18)] + list(weights)).reshape((9,9)).T
            A = [0 if w == 0 else 1 for w in (W+Fixed).flatten()]
            G = nx.Graph(np.array(A).reshape((9,9)))

    # Determining weight types
    W1 = [w_0_2, w_0_7, w_2_2, w_2_6, w_8_2, w_4_2]
    W2 = [w for w in W[:,4:].flatten() if type(w) != int]
    W3 = [w for w in W[:,2:4].flatten() if type(w) != int]

    W += Fixed
    return sp.Matrix(W), W1, W2, W3


### ======================
### Defining determinants
### ======================

print("Defining theoretical framework... [1/4]")

x_0 = 1
x_1, x_2, x_3, x_4, x_5, x_7, x_8, x_9 = sp.symbols(
    'x_1 x_2 x_3 x_4 x_5 x_7 x_8 x_9', real=True
)
X = [x_0, x_1, x_2, x_3, x_4, x_5, x_7, x_8, x_9]

### ======================
### Defining weights
### ======================

(w_0_2, w_0_7, w_2_2, w_2_6, w_0_3, w_0_4, w_0_5, w_0_8, w_0_9, w_1_3, w_1_4,
w_1_5, w_1_8, w_2_3, w_2_4, w_2_8, w_5_4, w_7_4, w_7_8, w_8_3, w_8_4, w_9_5,
w_1_2, w_3_1, w_4_2, w_5_1, w_7_1, w_8_1, w_9_1, w_8_2) = sp.symbols('w_0_2 \
w_0_7 w_2_2 w_2_6 w_0_3 w_0_4 w_0_5 w_0_8 w_0_9 w_1_3 w_1_4 w_1_5 w_1_8 w_2_3 \
w_2_4 w_2_8 w_5_4 w_7_4 w_7_8 w_8_3 w_8_4 w_9_5 w_1_2 w_3_1 w_4_2 w_5_1 w_7_1 \
w_8_1 w_9_1 w_8_2', real=True)

W_expert = np.array([
    [ 0,      0,      0,             0,      0,      0,       0,      0,     0],
    [ 0,      0,  w_7_1,             0,      0,  w_7_4,       0,  w_7_8,     0],
    [ 0,      0,      0,         w_1_2,  w_1_3,  w_1_4,   w_1_5,  w_1_8,     0],
    [ 0,      0,      0,             0,  w_2_3,  w_2_4,       0,  w_2_8,     0],
    [ 0,      0,  w_3_1,             0,      0,      0,       0,      0,     0],
    [ 0,      0,      0,             0,      0,      0,       0,      0,     0],
    [ 0,      0,  w_5_1,             0,      0,  w_5_4,       0,      0,     0],
    [ 0,      0,  w_8_1,             0,  w_8_3,  w_8_4,       0,      0,     0],
    [ 0,      0,  w_9_1,             0,      0,      0,   w_9_5,      0,     0]
])

Fixed = np.array([
    [ 1, -w_0_7,      0,  -w_4_2*w_0_2,  w_0_3,  w_0_4,   w_0_5,  w_0_8, w_0_9],
    [ 0,      0,      0,             0,      0,      0,       0,      0,     0],
    [ 0,      0,      1,             0,      0,      0,       0,      0,     0],
    [ 0,  w_2_6,      0, 1-w_4_2*w_2_2,      0,      0,       0,      0,     0],
    [ 0,      0,      0,             0,      0,      0,       0,      0,     0],
    [ 0,      0,      0,     0.9*w_4_2,      0,      0,       0,      0,     0],
    [ 0,      0,      0,             0,      0,      0,       0,      0,     0],
    [ 0,      0,      0,  -w_4_2*w_8_2,      0,      0,       0,      0,     0],
    [ 0,      0,      0,             0,      0,      0,       0,      0,     0]
])

### ======================
### Defining weight types
### ======================

W, W1, W2, W3 = getWeightMatrix(W_expert, Fixed, True)

### ======================
### Defining stock time development
### ======================

print("Defining finite difference approximation... [2/4]")

x = sp.Matrix(X)
x_t1 = W.T*x
x_t2 = W.T*W.T*x
s = sp.Matrix(x[2:4])
s_t1 = sp.Matrix(x_t1[2:4])
s_t2 = sp.Matrix(x_t2[2:4])

### ======================
### Defining first and second order time derivative
### ======================

ds_dt = sp.simplify(s_t1 - s)
d2s_dt2 = sp.simplify(1/2*(s_t2-2*s_t1+s))

### ======================
### Making error term, gradient and Hessian
### ======================

print("Defining error, gradient and Hessian... [3/4]")

alpha = 1
ds_dt_abs = sp.Matrix([sp.Abs(d) for d in ds_dt])
vec = ds_dt_abs - alpha*d2s_dt2
E_n = sp.exp(-vec.dot(vec))
grad_n = sp.Matrix(Derive(E_n, W3))
hess_n = Hessian(E_n, W3)

### ======================
### Making numerically efficient functions
### ======================

print("Lambdifying equations... [4/4]")
symbs = X[1:] + W1 + W2 + W3
num_error = sp.lambdify(symbs, E_n, "numpy")
num_gradient = sp.lambdify(symbs, grad_n, "numpy")
num_hessian = sp.lambdify(symbs, hess_n, "numpy")
