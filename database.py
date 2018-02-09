import numpy as np
import pandas as pd
from scipy.odr import *

VARIABLE_ENERGY_EXPENDITURE = 10
THERMOGENESYS = 1/7700 # kilos weight gain per calorie

df1 = pd.read_csv('data.csv', decimal=',', delimiter=';').set_index('Heliusnr')
df2 = pd.read_csv(
        'energy_intake.csv', decimal=',', delimiter=';'
        ).set_index('Heliusnr')
df3 = pd.read_csv(
        'squash_data.csv', decimal=',', delimiter=';'
        ).set_index('Heliusnr')
df = df1.join(df2)

### ==================
### Defining functions
### ==================

def reformat(l, reverse=False):
    """
    Identifies missing data and reverses high and low data values when 'reverse'
    """
    if reverse:
        m = max(l)
        return np.array([m-i if i != -1 else np.nan for i in l])
    return np.array([i if i != -1 else np.nan for i in l])

def toFloat(l):
    """
    Changes comma-seperated strings from first list to dot-seperated strings
    Changes strings to floats
    """
    new = l.str.replace(',', '.')
    new = pd.to_numeric(new, errors='coerce')
    return new

### ==============
### Defining data
### ==============

""" Ethnicity """
ethnicities = reformat(df["H1_EtnTotaal"])

""" State determinants """
stress = df["H1_PsychStress"]
weight = (df["H1_LO_GemGewicht"], df["H1_Gewicht"])
sleep = df["H1_SlaapInUren"]
energy_intake = df["ENKcal_Sum"] # QUITE A LOT OF DATA MISSING FOR THIS
income = df["inkomen"].apply(lambda x: x/365.0)
# perceived_fatness = df["H1_MeningGw"]
max_exercise = max(df["H1_Squash_totmwk"])
exercise = df["H1_Squash_totmwk"].apply(lambda x: x*960/max_exercise) # IN MINUTES PER DAY
discrimination = df["H1_Discr_meanscore"]

""" Weight values """
sex = df["H1_geslacht"] - 1
age = df["H1_lft"]
length = (df["H1_LO_GemLengte"], df["H1_Lengte"])
bmi_picture = df["H1_LichGelijk_unjumbled"]
ideal_body_image = df["H1_LichWens_unjumbled"]

### ======================
### Missing weight and length measurement are filled with self-reported value
### ======================

temp_weight = toFloat(weight[0])
repl_weight = weight[1]
weight = temp_weight.fillna(repl_weight)

temp_length = toFloat(length[0])
repl_length = length[1]
length = temp_length.fillna(repl_length)
inv_length = length.apply(lambda x: 1/((.01*x)**2)) # in 1/m^2, for BMI

### ======================
### BMI is calculated
### ======================

bmi = weight*inv_length

### ======================
### Moving to numpy data structure
### ======================

pd_determinants = [
    stress, weight, sleep, energy_intake, income,
    bmi, exercise, discrimination
    ]
should_reverse = [0, 0, 0, 0, 0, 0, 0, 0]
determinants = np.array([
    reformat(d, r) for d, r in zip(pd_determinants, should_reverse)
    ])

pd_weight_values = [sex, age, length, inv_length, bmi_picture, ideal_body_image]
should_reverse = [0, 0, 0, 0, 0, 0]
weight_values = np.array([
    reformat(a, r) for a, r in zip(pd_weight_values, should_reverse)
    ])

### ======================
### Removing incomplete data entries
### ======================

data = np.vstack([determinants, weight_values])
totalPoints = len(df)
noData = np.zeros(totalPoints)
for n, d in enumerate(data):
    h = d.copy()
    h[~np.isnan(h)] = 0
    h[np.isnan(h)] = 1
    noData += h
noData[noData>1] = 1

def r(l, select):
    return np.array([i for n, i in enumerate(l) if not select[n]])

ethnicities = r(ethnicities, noData)

stress = r(data[0], noData)
weight = r(data[1], noData)
sleep = r(data[2], noData)
energy_intake = r(data[3], noData)
income = r(data[4], noData)
bmi = r(data[5], noData)
exercise = r(data[6], noData)
discrimination = r(data[7], noData)

sex = r(data[8], noData)
age = r(data[9], noData)
length = r(data[10], noData)
inv_length = r(data[11], noData)
bmi_picture = r(data[12], noData)
ideal_body_image = r(data[13], noData)

### ======================
### Fitting BMI's to self-reported antropometry picture
### Using fit to determine ideal BMI
### ======================

def f(p, q):
    dummy_q = np.vstack(([1 for n in range(len(q))],q))
    return np.dot(dummy_q.T, p)

def fitBMI(x0, x1):
    input_data = x0
    output_data = x1

    # Set up ODR with the model and data.
    data = RealData(input_data, output_data)
    odr = ODR(data, Model(f), beta0=[0, 0])

    # Run the regression.
    return odr.run().beta

p = fitBMI(bmi_picture, bmi)
ideal_body_image = f(p, ideal_body_image)
perceived_fatness = bmi - ideal_body_image

### ======================
### Determining squash-energy-expenditure
### ======================

squash_p = df3['H1_Squash_totmwk']
AEE_p = df3['AEE_mean']
squash, AEE = [], []
for s, a in zip(squash_p, AEE_p):
    if s != ' ' and s != '0':
        squash.append(float(s))
        AEE.append(a)

cal_min = np.mean(np.array(AEE)/np.array(squash))
# print(np.mean(AEE/squash))

### ======================
### Predicting energy expenditure and making values callable
### ======================

variables_data = np.array([
        perceived_fatness, stress, weight, sleep, energy_intake, income,
        exercise, discrimination
        ])

constant_rest_energy = 6.25*length - 5*age - 166*sex + 5
variable_rest_energy = [VARIABLE_ENERGY_EXPENDITURE for i in range(len(noData[noData==0]))]
AEE = [cal_min for i in range(len(noData[noData==0]))]
thermogenesys = [THERMOGENESYS for i in range(len(noData[noData==0]))]
weights_data = np.array([
    constant_rest_energy, ideal_body_image, variable_rest_energy,
    inv_length, AEE, thermogenesys
    ])

def selectOnEthnicity(eth):
    """
    Selects all data for one of three ethnic groups

    Input:   One of three strings, 'NL', 'HIND' or 'MAROK'
    Returns: Tuple of two numpy arrays
             One array with all data on the variables
             One array with all data on some of the weights
    """
    ethnicDict = {'NL':1, 'HIND':2, 'MAROK':8}
    num = ethnicDict[eth]

    select = [0 if e == num else 1 for e in database.ethnicities]
    return (r(database.variables_data.T, select),
            r(database.weights_data.T, select))

def selectRandom(amount):
    """
    Selects all data for a given number of participants, selected at random.

    Input:   The amount of participants that should be returned
    Returns: Tuple of two numpy arrays
             One array with all data on the variables
             One array with all data on some of the weights
    """
    select = np.zeros(nrPoints)
    select[amount:] = 1
    np.random.shuffle(select)

    return (r(database.variables_data.T, select),
            r(database.weights_data.T, select))
