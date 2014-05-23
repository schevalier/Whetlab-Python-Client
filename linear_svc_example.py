# This example demonstrates the usage of spearmint within the context
# of sklearn.
# Here we train a linear support vector machine on the MNIST dataset,
# We use whetlab to choose the value of the SVM regularization term
# C, avoiding a much more wasteful grid search.
import whetlab
import numpy as np

# Define parameters to optimize
parameters = { 'C':{'type':'float', 'min':1.0, 'max':1000.0, 'size':1}}
outcome = {'name':'Classification accuracy', 'type':'float'}
name = 'SVM on MNIST'
description = 'Training an SVM using the sklearn library on the MNIST dataset'
access_token = '' # PUT VALID ACCESS TOKEN HERE
scientist = whetlab.Experiment(name=name, description=description,
    access_token=access_token, parameters=parameters, outcome=outcome)

# Setup scikit-learn experiment
from sklearn import svm
from sklearn.datasets import fetch_mldata

# Download the mnist dataset to the current working directory
mnist = fetch_mldata('MNIST original', data_home='.')

order = np.random.permutation(60000)
train_set = [mnist.data[order[:5000],:], mnist.target[order[:5000]]]
valid_set = [mnist.data[50000:60000,:], mnist.target[50000:60000]]

for i in range(20):
    # Get suggested new experiment
    job = scientist.suggest()

    # Perform experiment
    learner = svm.LinearSVC(**job)
    learner.fit(*train_set)
    accuracy = learner.score(*valid_set)

    # Inform scientist about the outcome
    scientist.update(job,accuracy)
    scientist.report()

