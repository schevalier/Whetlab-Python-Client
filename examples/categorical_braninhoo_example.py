# In this example we will optimize the 'Braninhoo' optimization benchmark with a small twist to 
# demonstrate how to set up a categorical variable.  There is also a constraint on the function.
import whetlab
import numpy as np

# Define parameters to optimize
parameters = { 'X' : {'type':'float','min':0,'max':15,'size':1},
               'Y' : {'type':'float','min':-5,'max':10,'size':1},
               'Z' : {'type': 'enum', 'options': ['bad','Good!','OK']}}

access_token = None # Either replace this with your access token or put it in your ~/.whetlab file.
name = 'Categorical Braninhoo'
description = 'Optimize the categorical braninhoo optimization benchmark'
outcome = {'name':'Negative Categorical Braninhoo output', 'type':'float'}
scientist = whetlab.Experiment(name=name, access_token=access_token, description=description, parameters=parameters, outcome=outcome)

# Braninhoo function
def categorical_braninhoo(X,Y,Z):
    if X > 10:
        return np.nan

    Z = 1 if Z == 'Good!' else 2 if Z == 'OK' else 3
    return np.square(Y - (5.1/(4*np.square(np.pi)))*np.square(X) + (5/np.pi)*X - 6) + 10*(1-(1./(8*np.pi)))*np.cos(X) + 10*Z;

for i in range(10000):
    # Get suggested new experiment
    job = scientist.suggest()

    # Perform experiment
    print job
    outcome = -categorical_braninhoo(**job)
    print outcome

    # Inform scientist about the outcome
    scientist.update(job,outcome)

