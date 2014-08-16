import whetlab
import numpy as np

# Define parameters to optimize
parameters = { 'X' : {'type':'integer','min':0,'max':1,'size':20}}
name = 'Boltzmann Machine'
description = 'Optimize a Boltzmann distribution.'
outcome = {'name':'Function value', 'type':'float'}
scientist = whetlab.Experiment(name=name, description=description,
                               parameters=parameters, outcome=outcome)

np.random.seed(1)
W = np.random.randn(20,20)
W = 0.5*(W + W.T)

# Quadratic function
def boltzmann_goodness(X):
    X = np.array(X)
    return X.T.dot(W).dot(X)

def main():
    for i in range(10000):
        # Get suggested new experiment
        job = scientist.suggest()

        # Perform experiment
        print job
        outcome = boltzmann_goodness(**job)
        print i, outcome


        # Inform scientist about the outcome
        scientist.update(job, outcome)
        #scientist.report()

if __name__ == '__main__':
    main()
