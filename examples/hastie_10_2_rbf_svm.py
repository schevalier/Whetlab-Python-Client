# In this example we optimize the 2-D Hastie problem from sklearn with a support vector machine.
import whetlab

# Define parameters to optimize
parameters = {'C':{'type':'float', 'min':1.0, 'max':1000.0, 'size':1},
              'gamma':{'type':'float', 'min':0.0, 'max':1.0, 'size':1}}
outcome = {'name':'Classification accuracy', 'type':'float'}
name = 'sklearn SVM'
access_token = None # Either replace this with your access token or put it in your ~/.whetlab file.
description = 'Training an SVM using the sklearn library'
scientist = whetlab.Experiment(name=name, access_token=access_token, description=description,
                               parameters=parameters, outcome=outcome)

# Setup scikit-learn experiment
from sklearn import svm, cross_validation
from sklearn.datasets import make_hastie_10_2
data = make_hastie_10_2(1000, 1234)

n_iterations = 20
for i in range(n_iterations):
    # Get suggested new experiment
    job = scientist.suggest()

    # Perform experiment using 5 fold cross validation.
    learner = svm.SVC(kernel='rbf', **job)
    accuracy = cross_validation.cross_val_score(
            learner, data[0], data[1], cv=5).mean()

    # Inform scientist about the outcome
    scientist.update(job, accuracy)
    scientist.report()
