# Import dependencies
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
# Import own code
import database, model, fit, validation

shuffle = True
nrRuns = 100
ethnicities_list = ["NL", "MAROK", "HIND"]

print("Defining symbolic framework... [1/7]")

### ======================
### Defining determinants
### ======================

x_0 = 1
x_1, x_2, x_3, x_4, x_5, x_7, x_8, x_9 = sp.symbols(
    'x_1 x_2 x_3 x_4 x_5 x_7 x_8 x_9', real=True
)
x_sym = sp.Matrix([x_0, x_1, x_2, x_3, x_4, x_5, x_7, x_8, x_9])

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

W_fixed = np.array([
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


if __name__ == '__main__':
    sensitivities, points = [], []
    for n in range(nrRuns):
        print("\n=======================================")
        print("Moving on to the %dth model simulation"%(n+1))
        print("=======================================")
        #Defining weight types
        W_sym, w1_sym, w2_sym, w3_sym, getL_n, getNewX = model.getWeightMatrix(
            W_expert, W_fixed, shuffle
            )

        sensitivities = []
        for n, eth in enumerate(ethnicities_list):
            print("\nNow computing minimum for",eth)
            print("----------------------------")
            print("Fitting auxiliary weights... [5/7]")
            (X_data, W1_data) = database.selectOnEthnicity(eth)
            w2_num = fit.Auxiliaries(X_data, W_sym)

            print("Fitting stock weights... [6/7]")
            w3_init = [0 for w in w3_sym]
            w3_num = fit.Stocks(X_data, W1_data, w2_num, getL_n, w3_init)

            print("Doing sensitivity analysis... [7/7]")
            sensitivities.append(
                validation.getSensitivity(W_sym, X_data, W1_data, w2_num, w3_num)
                )
        points.append(validation.getPoints(sensitivities))

    np.savetxt('results/points4.txt', points)
    plt.hist(points)
    plt.show()
