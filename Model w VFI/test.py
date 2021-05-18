
# load general packages
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('seaborn-whitegrid')
import pandas as pd

# load modules related to this exercise
import model_project as model

par = model.setup()

### children probabilities ###

# 1. settings
age_fer = 45 # maximum age of fertility
T = 35       # maximum periods from age_min needed to solve the model
num_n = 3    # maximum number of children

# 2. Child arrival probabilities
# i. allocate memory
shape = (num_n+1,T+1)
p = np.nan + np.zeros(shape)

# ii. load calibrations
birth_df = pd.read_excel('cali_birth_prob.xls') 
birth_df = birth_df.groupby(['nkids','age']).mean().reset_index()

# iii. Pick out relevant based on age and number of children
age_grid = np.array([age for age in range(par.age_min,par.age_min+T+1)])
for n in range(num_n+1):
    for iage,age in enumerate(age_grid):
        p[n,iage] = birth_df.loc[ (birth_df['age']==age) & (birth_df['nkids']==n) ,['birth']].to_numpy()

        if (age>age_fer) or (n==(num_n)):
            p[n,iage] = 0.0

# weighting function
par.theta0 = 0
par.theta1 = 0.2
par.N = 2

par = model.create_grids(par)
sol = model.solve(par)