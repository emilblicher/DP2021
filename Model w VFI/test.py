
# load general packages
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('seaborn-whitegrid')

# load modules related to this exercise
import model_project as model

par = model.setup()

par.theta0 = 0
par.theta1 = 0.2
par.N = 2

par = model.create_grids(par)
sol = model.solve(par)

util(10,10,par)