import whetlab
import numpy as np

# Define parameters to optimize
parameters = { 'X' : {'type':'float','min':0,'max':15,'size':1},
               'Y' : {'type':'float','min':-5,'max':10,'size':1}}
access_token = ''
name = 'Braninhoo'
description = 'Optimize the braninhoo optimization benchmark'
outcome = {'name':'Negative Braninhoo output', 'type':'float'}
scientist = whetlab.Experiment(name=name, description=description,
    access_token=access_token, parameters=parameters, outcome=outcome)

# Braninhoo function
def braninhoo(X,Y):
    return np.square(Y - (5.1/(4*np.square(np.pi)))*np.square(X) + (5/np.pi)*X - 6) + 10*(1-(1./(8*np.pi)))*np.cos(X) + 10;

for i in range(20):
    # Get suggested new experiment
    job = scientist.suggest()

    # Perform experiment
    print job
    outcome = -braninhoo(**job)

    # Inform scientist about the outcome
    scientist.update(job,outcome)
    scientist.report()

