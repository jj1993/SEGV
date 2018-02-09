import numpy as np

timesteps = 10
change = 1.1
weight, sleep, income, exercise, discrimination = 3, 4, 6, 7, 8

def getDefaultX(x, W_num, timesteps):
    x = np.hstack((1,x))
    for j in range(timesteps):
        x = np.dot(W_num.T, x)
    return x

def getInterventionX(x, W_num, timesteps, i):
    x = np.hstack((1,x))
    for j in range(timesteps):
        x[i] *= change
        x = np.dot(W_num.T, x)
    return x

def getSensitivity(getW_num, X_data, W1_data, w2_num, w3_num):
    sensitivity, X_default  = [], []

    for x, w1 in zip(X_data, W1_data):
        W_num = getW_num(*x.T, *w1.T, *w2_num, *w3_num)
        X_default.append(getDefaultX(x, W_num, timesteps))
    X_default = np.array(X_default)

    # Sleep, income, exercise, discr to weight
    for i in [sleep, income, exercise, discrimination]:
        X_intervention = []
        for x, w1 in zip(X_data, W1_data):
            W_num = getW_num(*x.T, *w1.T, *w2_num, *w3_num)
            X_intervention.append(getInterventionX(x, W_num, timesteps, i))
        X_intervention = np.array(X_intervention)

        sensitivity.append(np.mean(
                X_intervention.T[weight] - X_default.T[weight]
                ))

    return np.array(sensitivity)

def getPoints(sensitivities):
    points = 0

    test1, test2, test3, test4 = np.array(sensitivities).T
    print(np.array(sensitivities))
    # First test, sleep
    points += len(test1[test1 < 0])
    print(1, points)

    # Second test, income
    these_points = len(test2[test2 < 0])
    print('2a', points+these_points)
    if these_points == 3:
        if (test2[0] < test2[1]) and (test2[0] < test2[2]):
            these_points += 2
    points += these_points
    print('2b', points)

    # Third test, exercise (same order magnitude)
    ex1, ex2, ex3 = abs(test3)
    if (ex1/2 <= ex2 and ex2 <= ex1*2)  \
    and (ex1/2 <= ex3 and ex3 <= ex1*2)  \
    and (ex2/2 <= ex3 and ex3 <= ex2*2) :
        points += 3
    print(3, points)

    # Fourth test, rascism
    points += len(test4[test4 > 0])
    print(4, points)

    return points
