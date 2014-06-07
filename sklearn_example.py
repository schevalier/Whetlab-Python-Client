import whetlab

# Define parameters to optimize
parameters = {'C':{'type':'float', 'min':1.0, 'max':1000.0, 'size':1},
              'gamma':{'type':'float', 'min':0.0, 'max':1.0, 'size':1}}
outcome = {'name':'Classification accuracy', 'type':'float'}
name = 'sklearn SVM'
description = 'Training an SVM using the sklearn library'
access_token = 'f5f453f8-e38e-419f-81a1-14e674b81000' # PUT VALID ACCESS TOKEN HERE
scientist = whetlab.Experiment(name=name, description=description,
        access_token=access_token, parameters=parameters, outcome=outcome)

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
