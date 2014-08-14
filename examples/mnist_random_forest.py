# This example demonstrates the usage of spearmint within the context
# of sklearn.
# Here we train a random forest classifier on the MNIST dataset.
import whetlab
import numpy as np

# Define parameters to optimize
parameters = { 'n_estimators':{'type':'integer', 'min':2, 'max':100, 'size':1},
               'max_depth':{'type':'integer', 'min':1, 'max':20, 'size':1}}
outcome = {'name':'Classification accuracy', 'type':'float'}
name = 'Random Forest'
description = 'Training a random forest on the MNIST dataset using the sklearn library'
access_token = None # PUT VALID ACCESS TOKEN HERE OR IN YOUR ~/.whetlab FILE
scientist = whetlab.Experiment(name=name, description=description,
    access_token=access_token, parameters=parameters, outcome=outcome)

# Setup scikit-learn experiment
from sklearn.datasets import fetch_mldata
from sklearn.ensemble import RandomForestClassifier

# Download the mnist dataset to the current working directory
mnist = fetch_mldata('MNIST original', data_home='.')

order = np.random.permutation(60000)
train_set = [mnist.data[order[:50000],:], mnist.target[order[:50000]]]
valid_set = [mnist.data[order[50000:60000],:], mnist.target[order[50000:60000]]]

for i in range(20):
    # Get suggested new experiment
    job = scientist.suggest()

    # Perform experiment
    learner = RandomForestClassifier(**job)
    learner.fit(*train_set)
    accuracy = learner.score(*valid_set)

    # Inform scientist about the outcome
    scientist.update(job,accuracy)
    scientist.report()

