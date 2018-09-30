import numpy as np

timesteps = 30
change = 1.1
(fatness, stress, weight, sleep, energy_intake, income, exercise,
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

def getSensitivity(getW_num, X_data, W1_data, w2_num, w3_num, w4_num):
    sensitivity, X_default  = [], []

    for x, w1 in zip(X_data, W1_data):
        W_num = getW_num(*x.T, *w1.T, *w2_num, *w3_num, *w4_num)
        X_default.append(getDefaultX(x, W_num, timesteps))
    X_default = np.array(X_default)

    # Sleep, income, exercise, discr to weight
    for intervention in [sleep, income, fatness, exercise, discrimination]:
        X_intervention = []
        for x, w1 in zip(X_data, W1_data):
            W_num = getW_num(*x.T, *w1.T, *w2_num, *w3_num, *w4_num)
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
        ### This check got taken out because the data disagrees with the literature
        # if intervention == fatness:
        #     for measure in [stress]:
        #         sensitivity.append(
        #             getDiff(X_intervention, X_default, measure)
        #         )

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
    points, bonuspoints = 0, 0

    for nr, test in enumerate(np.array(sensitivities).T):
        nr += 1

        # Sleep -> stress, weight, energy intake
        if nr <= 3:
            points += len(test[test < 0])
        # Income -> weight
        if nr == 4:
            points += len(test[test < 0])
            if len(test[test < 0]) == 3:
                if (test[0] > test[1]) and (test[0] > test[2]):
                    bonuspoints += 2
        ### This check got taken out because the data disagrees with the literature
        # # Fatness -> Stress
        # if nr == 5:
        #     points += len(test[test > 0])
        #     print('5a', points)
        #     if len(test[test > 0]) == 3:
        #         if (test[0] > test[1]) and (test[0] > test[2]):
        #             points += 2
        #     print('5b', points)
        # Exercise -> Weight
        if nr == 5:
            points += len(test[test < 0])
            if len(test[test < 0]) == 3:
                ex1, ex2, ex3 = test
                if (ex1/4 <= ex2 and ex2 <= ex1*4)  \
                and (ex1/4 <= ex3 and ex3 <= ex1*4)  \
                and (ex2/4 <= ex3 and ex3 <= ex2*4) :
                    bonuspoints += 2
        # Discrimination -> Stress, weight, energy intake
        if nr >= 6:
            points += len(test[test > 0])

    return (points, bonuspoints)

def checkPoints(sensitivities):
    points = np.zeros((8,4))
    for nr, test in enumerate(np.array(sensitivities).T):

        # Sleep -> stress, weight, energy intake
        if nr <= 2:
            test[test>0] = 1
            test[test<0] = 0
            points[nr,:3] = 1 - test
        # Income -> weight
        if nr == 3:
            if len(test[test < 0]) == 3:
                if (test[0] > test[1]) and (test[0] > test[2]):
                    points[nr, 3] = 2
            test[test>0] = 1
            test[test<0] = 0
            points[nr,:3] = 1 - test
        ### This check got taken out because the data disagrees with the literature
        # # Fatness -> Stress
        # if nr == 4:
        #     points[nr] = len(test[test > 0])
        #     if len(test[test > 0]) == 3:
        #         if (test[0] > test[1]) and (test[0] > test[2]):
        #             points[nr] += 2
        # Exercise -> Weight
        if nr == 4:
            if len(test[test < 0]) == 3:
                ex1, ex2, ex3 = test
                if (ex1/4 <= ex2 and ex2 <= ex1*4)  \
                and (ex1/4 <= ex3 and ex3 <= ex1*4)  \
                and (ex2/4 <= ex3 and ex3 <= ex2*4) :
                    points[nr, 3] = 2
            test[test>0] = 1
            test[test<0] = 0
            points[nr,:3] = 1 - test
        # Discrimination -> Stress, weight, energy intake
        if nr >= 5:
            test[test>0] = 1
            test[test<0] = 0
            points[nr,:3] = test

    return points
