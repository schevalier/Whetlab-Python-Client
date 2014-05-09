Python Client
=============

Here's an example of optimizing the Braninhoo function using spearmint:

    import spearmint_client
    import numpy as np
    
    # Define parameters to optimize
    parameters = { 'X' : {'type':'float','min':0,'max':15,'size':1},
                   'Y' : {'type':'float','min':-5,'max':10,'size':1}}
    
    outcome = {'name':'Braninhoo output', 'type':'float'}
    scientist = spearmint_client.Experiment(parameters, outcome)
    
    # Braninhoo function
    def braninhoo(X,Y):
        return np.square(Y - (5.1/(4*np.square(np.pi)))*np.square(X) + (5/np.pi)*X - 6) + 10*(1-(1./(8*np.pi)))*np.cos(X) + 10;
    
    for i in range(20):
        # Get suggested new experiment
        job = scientist.suggest()
    
        # Perform experiment
        outcome = braninhoo(**job)
    
        # Inform scientist about the outcome
        scientist.update(job,outcome)
        scientist.report()



Here's an example of using spearmint to optimize Sciki-learn's RBF kernel SVM:

    import spearmint_client
    
    # Define parameters to optimize
    parameters = { 'C':{'type':'float', 'min':1, 'max':1000, 'size':1},
                   'gamma':{'type':'float', 'min':0, 'max':1, 'size':1}}
    outcome = {'name':'Classification error', 'type':'float'}
    scientist = spearmint_client.Experiment(parameters, outcome)
    
    # Setup scikit-learn experiment
    from sklearn import svm
    from sklearn.datasets import make_hastie_10_2
    all_set = make_hastie_10_2(1000,1234)
    train_set = [ set[:800] for set in all_set]
    valid_set = [ set[800:] for set in all_set]
    
    for i in range(20):
        # Get suggested new experiment
        job = scientist.suggest()
    
        # Perform experiment
        learner = svm.SVC(kernel='rbf',**job)
        learner.fit(*train_set)
        error = 1-learner.score(*valid_set)
    
        # Inform scientist about the outcome
        scientist.update(job,error)
        scientist.report()

