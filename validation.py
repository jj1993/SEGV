import numpy as np

timesteps = 30
change = 1.1
(stress, weight, sleep, energy_intake, income, fatness, exercise,
discrimination) = [i+1 for i in range(8)]

def getDiff(x_int, x_def, measurable):
    return (
        np.mean(x_int, axis=0)[measurable] -
        np.mean(x_def, axis=0)[measurable]
    )

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
        # # Predict food intake through linear regression fit to use incomplete
        # # data for validation!
        # x[4] = getFood_num(*x.T, *w1.T, *w2_num, *w3_num)

        W_num = getW_num(*x.T, *w1.T, *w2_num, *w3_num)
        X_default.append(getDefaultX(x, W_num, timesteps))
    X_default = np.array(X_default)

    # Sleep, income, exercise, discr to weight
    for intervention in [sleep, income, fatness, exercise, discrimination]:
        X_intervention = []
        for x, w1 in zip(X_data, W1_data):
            W_num = getW_num(*x.T, *w1.T, *w2_num, *w3_num)
            X_intervention.append(
                getInterventionX(x, W_num, timesteps, intervention)
            )
        X_intervention = np.array(X_intervention)

        if intervention == sleep:
            for measure in [stress, weight, energy_intake]:
                sensitivity.append(
                    getDiff(X_intervention, X_default, measure)
                )

        if intervention == income:
            for measure in [weight]:
                sensitivity.append(
                    getDiff(X_intervention, X_default, measure)
                )

        if intervention == fatness:
            for measure in [stress]:
                sensitivity.append(
                    getDiff(X_intervention, X_default, measure)
                )

        if intervention == exercise:
            for measure in [weight]:
                sensitivity.append(
                    getDiff(X_intervention, X_default, measure)
                )

        if intervention == discrimination:
            for measure in [stress, weight, energy_intake]:
                sensitivity.append(
                    getDiff(X_intervention, X_default, measure)
                )

    return np.array(sensitivity)

def getPoints(sensitivities):
    points = 0
    print(np.array(sensitivities))

    for nr, test in enumerate(np.array(sensitivities).T):
        nr += 1

        # Sleep -> stress, weight, energy intake
        if nr <= 3:
            points += len(test[test < 0])
            print(nr, points)
        # Income -> weight
        if nr == 4:
            points += len(test[test < 0])
            print('4a', points)
            if len(test[test < 0]) == 3:
                if (test[0] > test[1]) and (test[0] > test[2]):
                    points += 2
            print('4b', points)
        # Fatness -> Stress
        if nr == 5:
            points += len(test[test > 0])
            print('5a', points)
            if len(test[test > 0]) == 3:
                if (test[0] > test[1]) and (test[0] > test[2]):
                    points += 2
            print('5b', points)
        # Exercise -> Weight
        if nr == 6:
            points += len(test[test < 0])
            print('6a', points)
            if len(test[test < 0]) == 3:
                ex1, ex2, ex3 = test
                if (ex1/4 <= ex2 and ex2 <= ex1*4)  \
                and (ex1/4 <= ex3 and ex3 <= ex1*4)  \
                and (ex2/4 <= ex3 and ex3 <= ex2*4) :
                    points += 2
            print('6b', points)
        # Discrimination -> Stress, weight, energy intake
        if nr >= 7:
            points += len(test[test > 0])
            print(nr, points)

    return points
