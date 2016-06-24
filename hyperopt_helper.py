#import pandas
#import ctn_benchmark
#import seaborn as sns

import numpy as np
#import matplotlib as mpl
#import matplotlib.pyplot as plt
from hyperopt import fmin, hp, tpe, Trials 
import pickle

# define the objective, in which we want the value of 
# curve_width - 10 and shift - 5 to be minimized

# The parameters that we want to change must be childhood, t_glasses_on, and 
# the intercept range (maybe??)
import owl_parameter_search_intercepts
def objective(x):
    vals = owl_parameter_search_intercepts.Owl().run(
        childhood = x['childhood'], t_glasses_on = x['t_glasses_on']
    )
    return {
        'loss': {
            abs(vals['curve_width'] - 10) + abs(vals['shift'] - 5),
        },
        'status': STATUS_OK,
    }
trials = Trials()
best = fmin(objective,
            space = {'childhood': hp.uniform('childhood', 10, 10000), 't_glasses_on': hp.uniform('t_glasses_on', 10, 10000)},
            algo = tpe.suggest,
            max_evals = 10,
            trials = trials
           )
pickle.dump({'Trials': trials, 'Best': best}, open ('owl_hyperopt_data', 'w'))


