# Import dependencies
import sympy as sp
import networkx as nx
import numpy as np
# from sympy.tensor.array import derive_by_array as Derive
# from sympy.matrices.dense import hessian as Hessian

def getWeightMatrix(W_expert, W_fixed, w1_sym, x_sym, shuffle):
    W = W_expert
    if shuffle:
        # Creating a valid shuffled graph
        weights = W.T.flatten()[18:]
        np.random.shuffle(weights)
        W = np.array([0 for i in range(18)] + list(weights)).reshape((9,9)).T
        A = [0 if w == 0 else 1 for w in (W + W_fixed).flatten()]
        G = nx.Graph(np.array(A).reshape((9,9)))

        # Checks that no weights overlap the literature weights and
        # that the graph is connected
        while sum(
            [not (type(w) == int or type(f) == int) for w, f in
             zip(W.flatten(), W_fixed.flatten())]
        ) != 0 or not nx.is_connected(G):
            np.random.shuffle(weights)
            W = np.array([0 for i in range(18)] + list(weights)).reshape((9,9)).T
            A = [0 if w == 0 else 1 for w in (W + W_fixed).flatten()]
            G = nx.Graph(np.array(A).reshape((9,9)))

    # Determining weight types
    w3_sym = [w for w in W[:,2:4].flatten() if type(w) != int]

    W += W_fixed
    w2_sym = []
    for c in W[:,4:].T:
        w2_sym += [w for w in c if type(w) != int]
    symbs = x_sym[1:] + w1_sym + w2_sym + w3_sym
    W = sp.Matrix(W)

    ### ======================
    ### Defining stock time development
    ### ======================

    print("Defining finite difference approximation... [2/7]")

    x_t1 = W.T*x_sym
    x_t2 = W.T*W.T*x_sym
    x_t3 = W.T*W.T*W.T*x_sym
    s = sp.Matrix(x_sym[2:4])
    s_t1 = sp.Matrix(x_t1[2:4])
    s_t2 = sp.Matrix(x_t2[2:4])

    # Derivative given by first order foreward finite difference equation
    # with second order accuracy
    ds_dt = 1/2*(-3*s + 4*s_t1 - s_t2)

    ### ======================
    ### Making error term, gradient and Hessian
    ### ======================

    print("Defining error term... [3/7]")

    vec = ds_dt
    p1 = 1/(1+vec.dot(vec))
    # grad_n = sp.Matrix(Derive(E_n, W3))
    # hess_n = Hessian(E_n, W3)

    # Making numerically efficient functions
    print("Lambdifying equation... [4/7]")
    getL_n = sp.lambdify(symbs, -sp.log(p1), "numpy")
    # getNewX = sp.lambdify(symbs, x_t3, "numpy")
    getW_num = sp.lambdify(symbs, W)

    return W, w1_sym, w2_sym, w3_sym, getL_n, getW_num
