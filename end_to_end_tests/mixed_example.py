# In this example we will optimize the 'Braninhoo' optimization benchmark with a small twist to 
# demonstrate how to set up a categorical variable.  There is also a constraint on the function.
import whetlab
import numpy as np

# Define parameters to optimize
parameters = { 'f1' : {'type':'float','min':-1.5,'max':10.1,'size':1},
               'f2' : {'type':'float','min':-6.7,'max':3.33,'size':2},
               'i1' : {'type':'integer','min':1,'max':4,'size':1},
               'i2' : {'type':'integer','min':-10,'max':-7,'size':3},
               'c1' : {'type': 'enum', 'options': ['a','b','c']},
               'c2' : {'type': 'enum', 'options': ['red','green','blue','yellow'], 'size': 2}}

access_token = None # Either replace this with your access token or put it in your ~/.whetlab file.
name = 'Mixed Type Example'
description = 'This function is just noisy, but has a lot of interesting input types.'
outcome = {'name':'Garbage'}
scientist = whetlab.Experiment(name=name, access_token=access_token, description=description, parameters=parameters, outcome=outcome)

# Braninhoo function
def garbage(**params):
    if np.random.rand() < 0.2:
      return np.nan
    else:
      return np.random.randn()

for i in range(10000):
    # Get suggested new experiment
    job = scientist.suggest()

    # Perform experiment
    print job
    outcome = -garbage(**job)
    print outcome

    # Inform scientist about the outcome
    scientist.update(job,outcome)

