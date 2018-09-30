# Import dependencies
import numpy as np
import networkx as nx
import sympy as sp
import matplotlib.pyplot as plt
# Import own code
import database, model, fit, validation, similarities
from sklearn.model_selection import KFold
import itertools
import pandas as pd
import sys

try: run_type = sys.argv[1]
except: raise ValueError("\n\nExecute python file as 'python main.py run_type'.\
\n Where run_type is 'expert', 'agnost' or 'alphas'.")
if not run_type in ["expert", "agnost", "alphas"]:
    raise ValueError("\n\nExecute python file as 'python main.py run_type'.\
\n Where run_type is 'expert', 'agnost' or 'alphas'.")
alpha = 2e-4 # Default alpha value
nrRuns = 50 # Number of agnost models or number of alpha values. Not used for expert model
if run_type == 'expert': nrRuns = 1
alphas = np.logspace(-5, -1, nrRuns) # Different alpha values for alpha run type
nrFolds = 3 # Number of folds for bootstrap of model scores
z = 7 # File name extention for saved data
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

(w_1_2, w_1_5,  w_1_7, w_2_3,  w_2_4,  w_2_5,   w_2_6,  w_2_7, w_3_4,  w_3_5,
w_3_7, w_4_2, w_6_2, w_6_5, w_7_2, w_7_4,  w_7_5, w_8_2, w_8_6, w_0_1, w_0_3,
w_0_4,  w_0_5, w_0_6, w_0_7, w_0_8, w_3_1, w_3_3, w_7_3, w_therm) = sp.symbols(
'w_1_2 w_1_5  w_1_7 w_2_3 w_2_4 w_2_5 w_2_6 w_2_7 w_3_4  w_3_5 w_3_7 w_4_2 \
w_6_2 w_6_5 w_7_2 w_7_4 w_7_5 w_8_2 w_8_6 w_0_1 w_0_3 w_0_4 w_0_5 w_0_6 w_0_7 \
w_0_8 w_3_1 w_3_3 w_7_3 w_therm', real=True)

w1_sym = [w_0_1, w_0_3, w_3_1, w_3_3, w_7_3, w_therm]

W_expert = np.array([
    [ 0,      0,      0,             0,      0,      0,       0,      0,     0],
    [ 0,      0,  w_1_2,             0,      0,  w_1_5,       0,  w_1_7,     0],
    [ 0,      0,      0,         w_2_3,  w_2_4,  w_2_5,   w_2_6,  w_2_7,     0],
    [ 0,      0,      0,             0,  w_3_4,  w_3_5,       0,  w_3_7,     0],
    [ 0,      0,  w_4_2,             0,      0,      0,       0,      0,     0],
    [ 0,      0,      0,             0,      0,      0,       0,      0,     0],
    [ 0,      0,  w_6_2,             0,      0,  w_6_5,       0,      0,     0],
    [ 0,      0,  w_7_2,             0,  w_7_4,  w_7_5,       0,      0,     0],
    [ 0,      0,  w_8_2,             0,      0,      0,   w_8_6,      0,     0]
])

W_fixed = np.array([
    [ 1, -w_0_1,      0,  -w_therm*w_0_3,  w_0_4,  w_0_5,   w_0_6,  w_0_7, w_0_8],
    [ 0,      0,      0,               0,      0,      0,       0,      0,     0],
    [ 0,      0,      1,               0,      0,      0,       0,      0,     0],
    [ 0,  w_3_1,      0, 1-w_therm*w_3_3,      0,      0,       0,      0,     0],
    [ 0,      0,      0,               0,      0,      0,       0,      0,     0],
    [ 0,      0,      0,     0.9*w_therm,      0,      0,       0,      0,     0],
    [ 0,      0,      0,               0,      0,      0,       0,      0,     0],
    [ 0,      0,      0,  -w_therm*w_7_3,      0,      0,       0,      0,     0],
    [ 0,      0,      0,               0,      0,      0,       0,      0,     0]
])

def makeBarPlot(data_mean, data_std):
    labels = ["Dutch", "Moroccan", "Hindustan", "Combinatory"]
    plt.figure(figsize=(10,3))
    for n, (mean, std) in enumerate(zip(data_mean, data_std)):
        xts = [i+.2*(n-1)-.1 for i in range(1,9)]
        mean = mean*.95+.05
        if n == 3: mean[[0,1,2,5,6,7]] = 0
        plt.bar(xts, mean, width=.2, yerr = std, align="center", label=labels[n])
    plt.xticks(range(1,9))
    plt.yticks((.05,1), (0, 1))
    plt.ylabel("Average scoring")
    plt.xlabel("Check number")
    plt.title("Average scoring of the expert model per ethnicity")
    plt.legend(loc=4)
    plt.tight_layout()
    plt.savefig("../figures/expert_scoring")
    plt.show()

if __name__ == '__main__':
    kf = KFold(n_splits=nrFolds, shuffle=False)
    points_list = np.empty((nrRuns, 2))
    similarity_list = np.empty((nrRuns, 1))
    model_list = np.empty((nrRuns, len(x_sym)**2))

    for n in range(nrRuns):
        print("\n=======================================")
        print("Simulating model run",n+1)
        print("=======================================")
        # Defining weight types
        if run_type == 'agnost': shuffle_weights = True
        else:               shuffle_weights = False
        W_sym, w1_sym, w2_sym, w3_sym, w4_sym, getNewStress, getNewWeight, \
        getW_num = model.getWeightMatrix(
                W_expert, W_fixed, w1_sym, x_sym, shuffle_weights
                )

        sensitivity_list = [[], [], []]
        mean_weights = [[], [], []]

        for m, eth in enumerate(ethnicities_list):
            print("\nNow computing minimum for",eth)
            print("----------------------------")
            (X_data, W1_data) = database.selectOnEthnicity(eth)

            these_weights = []
            for train, val in kf.split(X_data):
                X_train, W1_train = X_data[train], W1_data[train]
                X_val, W1_val = X_data[val], W1_data[val]

                print("Fitting auxiliary weights... [5/7]")
                w2_num = fit.Auxiliaries(X_train, W_sym)
                print("Fitting stock weights... [6/7]")
                if run_type == 'alphas': alpha = alphas[n]
                w3_num = fit.Stress(X_train, W1_train, w3_sym, getNewStress, alpha)
                w4_num = fit.Weight(X_train, W1_train, w4_sym, getNewWeight)

                print("Doing analysis... [7/7]")
                sensitivity_list[m].append(validation.getSensitivity(
                    getW_num, X_val, W1_val, w2_num, w3_num, w4_num
                    )
                )
                if run_type == 'expert':
                    these_weights.append(np.hstack((w2_num, w3_num, w4_num)))

            if run_type == 'expert':
                # Save the mean weights and data of the fitted model to excel
                print("Writing data to xls...")
                mean_weights[m] = np.mean(these_weights, axis=0)
                writer = pd.ExcelWriter('results/'+eth+'.xls')
                symbs = [str(s) for s in [*w2_sym, *w3_sym, *w4_sym]]
                df = pd.DataFrame(np.array(mean_weights[m])).transpose()
                df.columns = symbs
                df.to_excel(writer,'weights')

                symbs = [str(s) for s in [*x_sym[1:], *w1_sym]]
                data = np.hstack((X_data, W1_data))
                for l, d in enumerate(data):
                    df = pd.DataFrame(d).transpose()
                    df.columns = symbs
                    df.to_excel(writer,str(l))
                df = pd.DataFrame(np.mean(data, axis=0)).transpose()
                df.columns = symbs
                df.to_excel(writer,'mean')
                writer.save()

        # The average scoring and its standard deviation are calculated
        allPoints = np.array([
            validation.getPoints(d) for d in
            itertools.product(*sensitivity_list)
        ])
        points_mean = sum(np.mean(allPoints, axis=0))
        std = np.std(allPoints, axis=0)
        # The bonus points are calculated 27 times independently, the normal
        # points only 3 times. Thence the different devisions
        points_std = std[0]/np.sqrt(len(ethnicities_list)) \
                   + std[1]/len(ethnicities_list)
        print("Model points:")
        print(points_mean, " pm ", points_std)

        if run_type == 'expert':
            # The expert runtype looks in detail into the average score of each check
            data = [
                validation.checkPoints(d) for d in
                itertools.product(*sensitivity_list)
            ]
            data_mean = np.mean(data, axis=0).T
            data_mean[3,:] /= 2
            data_std = np.std(data, axis=0).T
            data_std[:3,:] /= np.sqrt(3)
            data_std[3,:] /= 3*2**2
            makeBarPlot(data_mean, data_std)

        if run_type == 'agnost' or run_type == 'alphas':
            # The agnost and alphas runtypes save the average points of each run
            points_list[n] = (points_mean, points_std)
            np.savetxt('results/points%d.txt'%z, points_list)

        if run_type == 'agnost':
            # The agnost runtype saves the similarities and weights matrix of each run
            similarity_list[n] = \
                similarities.getSimilarity(W_expert, W_sym, W_fixed)

            W = np.array(W_sym).flatten()
            W[W!=0]=1
            model_list[n] = np.array(W, dtype=int)

            np.savetxt('results/models%d.txt'%z, model_list)
            np.savetxt('results/similarities%d.txt'%z, similarity_list)
