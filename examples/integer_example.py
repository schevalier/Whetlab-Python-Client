import whetlab
import numpy as np

# Define parameters to optimize
parameters = { 'X' : {'type':'integer','min':0,'max':15,'size':1},
               'Y' : {'type':'integer','min':-5,'max':10,'size':1}}
name = 'Integer Example'
access_token = None # Either replace this with your access token or put it in your ~/.whetlab file.
description = 'Optimize a simple quadratic with integer variables.'
outcome = {'name':'Function value', 'type':'float'}
scientist = whetlab.Experiment(name=name, access_token=access_token, description=description,
                               parameters=parameters, outcome=outcome)

# Quadratic function
def f(X,Y):
    return -((X-3)**2 + (Y+1)**2)

for i in range(10000):
    # Get suggested new experiment
    job = scientist.suggest()

    # Perform experiment
    print job
    outcome = f(**job)
    print i, outcome


    # Inform scientist about the outcome
    scientist.update(job, outcome)
    #scientist.report()

