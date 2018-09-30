import numpy as np
import pandas as pd
from scipy.odr import *

THERMOGENESYS = 1/7700 # kilos weight gain per calorie, (Katan & Ludwig, 2010)

# HELIUS study
df1 = pd.read_csv('data/data.csv', decimal=',', delimiter=';').set_index('Heliusnr')
# Additional study on energy intake in calories done by Mary Nicolaou
# for participants of the HELIUS study
df2 = pd.read_csv(
        'data/energy_intake.csv', decimal=',', delimiter=';'
        ).set_index('Heliusnr')
# Additional study on calories burnt per SQUASH minute exercise
# for participants of the HELIUS study (Nicolaou et al., 2016)
df3 = pd.read_csv(
        'data/squash_data.csv', delimiter=','
        ).set_index('Heliusnr')
df = df1.join(df2)

### ==================
### Defining functions
### ==================

def reformat(l):
    """
    Identifies missing data
    """
    return np.array([i if i != -1 else np.nan for i in l])

def toFloat(l):
    """
    Changes comma-seperated strings from first list to dot-seperated strings
    Changes strings to floats
    """
    new = l.str.replace(',', '.')
    new = pd.to_numeric(new, errors='coerce')
    return new

def r(l, select):
    """
    Selects data points according to select array
    """
    return np.array([i for n, i in enumerate(l) if not select[n]])

def f(p, q):
    """
    Fitting function for linear BMI regression
    """
    dummy_q = np.vstack(([1 for n in range(len(q))],q))
    return np.dot(dummy_q.T, p)

def fitBMI(x0, x1):
    """
    Linear BMI regression on body-image pictures
    """
    input_data = x0
    output_data = x1

    # Set up ODR with the model and data.
    data = RealData(input_data, output_data)
    odr = ODR(data, Model(f), beta0=[0, 0])

    # Run the regression.
    return odr.run().beta

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
max_exercise = max(df["H1_Squash_totmwk"])
exercise = df["H1_Squash_totmwk"].apply(lambda x: x*960/max_exercise) # IN MINUTES PER DAY
discrimination = df["H1_Discr_meanscore"]

""" Weight values """
sex = df["H1_geslacht"]
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

pd_variables = [
    stress, weight, sleep, energy_intake, income,
    bmi, exercise, discrimination
    ]
variables = np.array([reformat(d) for d in pd_variables])

pd_weight_values = [sex, age, length, inv_length, bmi_picture, ideal_body_image]
weight_values = np.array([reformat(a) for a in pd_weight_values])

### ======================
### Removing incomplete data entries
### Seperating data that only misses calory intake data
### ======================

data = np.copy(np.vstack([variables, weight_values]))
totalPoints = len(df)
select = np.zeros(totalPoints)
for n, d in enumerate(data):
    h = d.copy()
    h[~np.isnan(h)] = 0
    h[np.isnan(h)] = 1
    select += h
select[select>1] = 1

ethn_data = r(ethnicities, select)

stress = r(data[0], select)
weight = r(data[1], select)
sleep = r(data[2], select)
energy_intake = r(data[3], select)
income = r(data[4], select)
bmi = r(data[5], select)
exercise = r(data[6], select)
discrimination = r(data[7], select)

sex = r(data[8], select)
age = r(data[9], select)
length = r(data[10], select)
inv_length = r(data[11], select)
bmi_picture = r(data[12], select)
ideal_body_image = r(data[13], select)

### ======================
### FITTING PERCEIVED FATNESS ON BMI UNITS
###
### 1) Fitting BMI's to self-reported antropometry picture
### 2) Using fit to determine ideal BMI
### 3) Perceived fatness is defined as the difference between 1) and 2)
### (Clearification: The numbers from 'bmi_picture' and 'ideal_body_image'
### are self-reported numbers selected on images from the HELIUS study)
### ======================

p = fitBMI(bmi_picture, bmi) # 1)
ideal_body_image = f(p, ideal_body_image) # 2)
perceived_fatness = bmi - ideal_body_image # 3)

### ======================
### Determining squash-energy-expenditure
### Additional study on calories burnt per SQUASH minute exercise
### for participants of the HELIUS study (Nicolaou et al., 2016)
### ======================

squash_p = df3['H1_Squash_totmwk']
AEE_p = df3['AEE_mean']
squash, AEE = [], []
for s, a in zip(squash_p, AEE_p):
    if s != ' ' and s != '0' and \
       a != ' ' and a != '0':
        squash.append(float(s))
        AEE.append(float(a.replace(',','.')))

cal_min = np.mean(np.array(AEE)/np.array(squash))
AEE = [cal_min for i in range(len(select[select==0]))]

### ======================
### Predicting energy expenditure and returning data
### ======================

# Energy expenditure in rest from Schofield equations
# "Human energy requirements" - Report of a Joint FAO/WHO/UNU Expert
# Consultation, Rome, 17–24 October 2001
male, female = 1, 2
constant_rest_energy, variable_rest_energy = [], []
for s, a in zip(sex, age):
    if s == male:
        if a <= 30:
            variable_rest_energy.append(15.057)
            constant_rest_energy.append(692.2)
        elif a <= 60:
            variable_rest_energy.append(11.472)
            constant_rest_energy.append(873.1)
        else:
            variable_rest_energy.append(11.711)
            constant_rest_energy.append(587.7)

    if s == female:
        if a <= 30:
            variable_rest_energy.append(14.818)
            constant_rest_energy.append(486.6)
        elif a <= 60:
            variable_rest_energy.append(8.126)
            constant_rest_energy.append(845.6)
        else:
            variable_rest_energy.append(9.082)
            constant_rest_energy.append(658.5)

# Defining cleaned data
variables_data = np.array([
        perceived_fatness, stress, weight, sleep, energy_intake, income,
        exercise, discrimination
        ])

thermogenesys = np.array([THERMOGENESYS for i in range(len(select[select==0]))])
weights_data = np.array([
    ideal_body_image, constant_rest_energy, inv_length, variable_rest_energy,
    AEE, thermogenesys
    ])

# Defining functions to request data from this module
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

    select = [0 if e == num else 1 for e in ethn_data]

    return (r(variables_data.T, select),
            r(weights_data.T, select))

def selectRandom(amount):
    """
    Selects all data for a given number of participants, selected at random.

    Input:   The amount of participants that should be returned
    Returns: Tuple of two numpy arrays
             One array with all data on the variables
             One array with all data on some of the weights
    """
    select = np.zeros(len(noData[noData==0]))
    select[amount:] = 1
    np.random.shuffle(select)

    np.random.shuffle(noFood_select)

    return (r(variables_data.T, select),
            r(weights_data.T, select))
