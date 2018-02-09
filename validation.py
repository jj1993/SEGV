def getSensitivity(W_sym, X_data, W1_data, w2_num, w3_num):
    sensitivity = []
    X_check = np.array([getNewX(*x.T, *w1.T, *w2_num, *w3_num).flatten()[1:] for x, w1 in zip(X_data, W1_data)])
    change = 1.5

    # Sleep, income, exercise, discr to weight
    for i in [3, 5, 6, 7]:
        X_test = np.copy(X_data)
        X_test.T[i] *= change
        X_new = np.array([getNewX(*x.T, *w1.T, *w2_num, *w3_num).flatten()[1:] for x, w1 in zip(X_test, W1_data)])
        sensitivity.append(np.mean(X_new.T[2] - X_check.T[2]))

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
