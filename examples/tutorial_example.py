
from sklearn.datasets import fetch_mldata
data_set = fetch_mldata('yahoo-web-directory-topics')

train_set = (data_set['data'][:1000],data_set['target'][:1000])
validation_set = (data_set['data'][1000:],data_set['target'][1000:])

parameters = { 'C':{'min':0.01, 'max':1000.0,'type':'float'},
               'degree':{'min':1, 'max':5,'type':'integer'}}
outcome = {'name':'Classification accuracy'}
access_token = None # Either replace this with your access token or put it in your ~/.whetlab file.

import whetlab

# First remove this experiment if it already exists.
try:
    whetlab.delete_experiment(access_token, "Web page classifier")
except:
    pass
scientist = whetlab.Experiment(access_token=access_token,
                               name="Web page classifier",
                               description="Training a polynomial kernel SVM to classify web pages.",
                               parameters=parameters,
                               outcome=outcome)

job = scientist.suggest()
from sklearn import svm
learner = svm.SVC(kernel='poly',**job)
learner.fit(*train_set)
accuracy = learner.score(*validation_set)
scientist.update(job,accuracy)
scientist.report()

n_iterations = 19
for i in range(n_iterations):
    job = scientist.suggest()
    learner = svm.SVC(kernel='poly',**job)
    learner.fit(*train_set)
    accuracy = learner.score(*validation_set)
    scientist.update(job,accuracy)
    scientist.report()

print scientist.best()

