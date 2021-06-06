
# load general packages
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('seaborn-whitegrid')
import pandas as pd

# load modules related to this exercise
import model_project as model

par = model.setup()

# weighting function
par.theta0 = 0
par.theta1 = 0.1

par = model.create_grids(par)
sol = model.solve(par)