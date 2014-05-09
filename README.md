Welcome to Whetlab!
===================

Whetlab automates the tuning of your favorite tool and optimizes its
performance.

What "tool" you ask? Well, it could be a lot of things.  It could be a
piece of software, whose performance is controlled by a few parameters
(such as a compression computer program or a machine learning
algorithm), or it could even be a complicated physical process (such
as the manufacture of a device or even a cooking recipe). As long as
your tool has a few knobs which you can crank up or down in order to
impact its performance, Whetlab can help you!

Whetlab works by suggesting tests you should run with your tool in
order to improve it. Once you have the result of these tests, you tell
Whetlab how they turned out and it will suggest new tests, and so on
until you're satisfied with the improved performance of your tool.

Installation Instructions
=========================

Installation is really simple.  Simply navigate to the directory containing whetlab.py and type:

    python setup.py install

Tutorial: Sharpening your tools with Whetlab
============================================

In this tutorial, we'll describe the functionalities of
Whetlab through a simple but important example application:
the tuning of a machine learning algorithm.

Machine learning algorithms are becoming ubiquitous in advanced
computer systems. Yet using them with success can require some
know-how. Specifically, each learning algorithm
requires the specification of hyper-parameters, which are
knobs that greatly impact the performance. Tuning these knobs is thus a
perfect problem for Whetlab to solve.

We'll also use the great machine learning library [scikit-learn](<http://scikit-learn.org/>)
library, which provides good implementations the most commonly used
learning algorithms.

Let's assume that we'd like to develop a classifier that can
automatically classify web pages according to the topic they discuss.
The first step is then to obtain some training data for this problem.
Luckily, scikit-learn provides us with simple functions for
downloading data sets from the [mldata.org](http://mldata.org/)
repository.  The reposity contains the [Yahoo! Web Directory Topics](http://mldata.org/repository/data/viewslug/yahoo-web-directory-topics/),
a data set of web pages<sup>1</sup> labeled with their Yahoo! directory topic
(Arts, Business, etc.).

The data set can be downloaded as follows:

    from sklearn.datasets import fetch_mldata
    data_set = fetch_mldata('yahoo-web-directory-topics')

In this case, ``data_set`` is a dictionary. It has a key ``'data'``
associated with a sparse matrix whose rows are the
web pages. It also has a key ``target`` corresponding to
a vector (Numpy 1D array) providing the class labels
of all web pages.

The next step is to split this data set into training and validation sets.
The data set contains a total of 1106 web pages, so we'll use 1000 for
training and the rest for searching over hyper-parameters: ::

    train_set = (data_set['data'][:1000],data_set['target'][:1000])
    validation_set = (data_set['data'][1000:],data_set['target'][1000:])

Next, we have to choose a learning algorithm to perform classification.
A popular choice is an SVM classifier, with a radial basis function (RBF) kernel<sup>2</sup>.
Its two most important hyper-parameters to tune are the regularization constant ``C``
and the lengthscale of the RBF kernel ``gamma``.

To inform Whetlab that we will be tuning with respect to these hyper-parameters,
we'll write down this information into a dictionary, as follows:

    parameters = { 'C':{'type':'float', 'min':1.0, 'max':1000.0, 'size':1},
                   'gamma':{'type':'float', 'min':0.0, 'max':1.0, 'size':1}}

In this dictionary, each key is a ``str`` corresponding to the name of
hyper-parameter. It is recommended make it identical to the corresponding
argument name that sciki-learn uses for it (we'll see later why). Associated
with each key, is a dictionary that provides information about the hyper-parameter.

As is probably obvious, the type of the hyper-parameter is specified by the key ``'type'``,
while ``'min'`` and ``'max'`` specifies the minimum and maximum values allowed for the
hyper-parameter. Finally, ``'size'`` specifies whether the hyper-parameter is a scalar
(size of 1) or a vector (size greater than 1).

We also need to tell Whetlab what we will be optimizing. In this case, we want
to minimize the validation set classification error, which we specify as follows:

    outcome = {'name':'Classification error', 'type':'float'}

Note that Whetlab always minimizes, so if one was interested to optimize a measure
of performance which increases with the quality of the solution, then you would
provide the negative of these measured outcomes.

We are now ready to start experimenting. First, we create a Whetlab experiment,
using the information about the hyper-parameters to tune and the type of outcome
to minimize: ::

    scientist = spearmint_client.Experiment(parameters, outcome)

We can now use ``scientist`` to suggest a first job to run:

    job = scientist.suggest()

Here, ``job`` is a dictionary, whose keys are the names of the hyper-parameters
and the associated values are suggested values to test. We can now instantiate
a scikit-learn SVM object and train it on our training set:

    from sklearn import svm
    learner = svm.SVC(kernel='rbf',**job)
    learner.fit(*train_set)

Notice that, since we have used names that match the
arguments of the constructor of the scikit-learn SVM object,
we can unpack the dictionary ``job`` as arguments to the
constructor by prefixing it with ``**``.

Once the SVM is trained, we can evaluate its performance on
the validation set and inform Whetlab of the outcome, using
the method ``update`` of ``scientist``:

    error = 1-learner.score(*valid_set)
    scientist.update(job,error)

Thanks to this information, Whetlab will be able to suggest
another promising job to run. Hence, with a simple ``for`` loop,
the process of tuning the SVM becomes:

    n_iterations = 20
    for i in range(n_iterations):
        job = scientist.suggest()
        learner = svm.SVC(kernel='rbf',**job)
        learner.fit(*train_set)
        error = 1-learner.score(*valid_set)
        scientist.update(job,error)

Once you're done tuning, we can simply ask ``scientist`` to provide you with
the best hyper-parameters found so far:

    best_job = scientist.best()

These are the hyper-parameter values you should be using to train your
final SVM classifier<sup>3</sup>.

<sup>1</sup>This data set is actually quite small for building a good classifier, it'll do for our purposes.

<sup>2</sup>For high-dimensional problems such as this one, an RBF kernel SVM might not be the best choice. However it's a popular choice in general, so we use it here.

<sup>3</sup>Simple trick to obtain better results: train the SVM classifier on all the data, i.e. the concatenation of the training and validation sets data.


Python Client
=============

Here's an example of using spearmint to optimize Sciki-learn's RBF kernel SVM:

    import whetlab
    
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

Here's an example of optimizing the Braninhoo function using spearmint:

    import whetlab
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

