# In this example we optimize a logistic regression on the popular MNIST digit recognition benchmark using the 'KAYAK'
# deep learning code base.
# To run this example, first get KAYAK from https://github.com/HIPS/Kayak. You can just run the command:
# 'git clone https://github.com/HIPS/Kayak.git' from this directory.

import sys
sys.path.extend(['Kayak', '..', 'Kayak/examples'])

import numpy as np
import numpy.random as npr
try:
    import kayak
except:
    print "This examples requires the Python library kayak (https://github.com/HIPS/Kayak)"
    sys.exit()

import data
import whetlab

num_folds = 5

# Here I define a nice little training function that takes inputs and targets.
def train(inputs, targets, batch_size, learn_rate, momentum, l1_weight, l2_weight, dropout, improvement_thresh):

    # Create a batcher object.
    batcher = kayak.Batcher(batch_size, inputs.shape[0])

    # Inputs and targets need access to the batcher.
    X    = kayak.Inputs(inputs, batcher)
    T    = kayak.Targets(targets, batcher)

    # Weights and biases, with random initializations.
    W    = kayak.Parameter( 0.1*npr.randn( inputs.shape[1], 10 ))
    B    = kayak.Parameter( 0.1*npr.randn(1,10) )

    # Nothing fancy here: inputs times weights, plus bias, then softmax.
    Y    = kayak.LogSoftMax( kayak.ElemAdd( kayak.MatMult(kayak.Dropout(X, dropout), W), B ) )

    # The training loss is negative multinomial log likelihood.
    loss = kayak.MatAdd(kayak.MatSum(kayak.LogMultinomialLoss(Y, T)),
                        kayak.L2Norm(W, l2_weight),
                        kayak.L1Norm(W, l1_weight))

    # Use momentum for the gradient-based optimization.
    mom_grad_W = np.zeros(W.shape())

    best_loss  = np.inf
    best_epoch = -1

    # Loop over epochs.
    for epoch in xrange(500):

        # Track the total loss and the overall gradient.
        total_loss   = 0.0
        total_grad_W = np.zeros(W.shape())

        # Loop over batches -- using batcher as iterator.
        for batch in batcher:

            # Compute the loss of this minibatch by asking the Kayak
            # object for its value and giving it reset=True.
            total_loss += loss.value(True)

            # Now ask the loss for its gradient in terms of the
            # weights and the biases -- the two things we're trying to
            # learn here.
            grad_W = loss.grad(W)
            grad_B = loss.grad(B)
            
            # Use momentum on the weight gradient.
            mom_grad_W = momentum*mom_grad_W + (1.0-momentum)*grad_W

            # Now make the actual parameter updates.
            W.add( -learn_rate * mom_grad_W )
            B.add( -learn_rate * grad_B )

            # Keep track of the gradient to see if we're converging.
            total_grad_W += grad_W

        print epoch, total_loss, np.sum(total_grad_W**2)

        if total_loss < best_loss:
            best_epoch = epoch
        else:
            if (epoch - best_epoch) > improvement_thresh:
                print "Has been %d epochs without improvement. Aborting." % (epoch-best_epoch)
                break

    # After we've trained, we return a sugary little function handle
    # that makes things easy.  Basically, what we're doing here is
    # handing the output object (not the loss!) a dictionary where the
    # key is the Kayak input object 'X' (that is the features being
    # used here for logistic regression) and the value in that
    # dictionary is being determined by the argument to the lambda
    # expression.  The point here is that we wind up with a function
    # handle the can be called with a numpy object and it produces the
    # target values for novel data, using the parameters we just learned.
    return lambda x: Y.value(True, inputs={ X: x })

def evaluate(batch_size, log10_lr, momentum, log10_l1, log10_l2, dropout, improvement_thresh):

    # Load in the MNIST data.
    train_images, train_labels, test_images, test_labels = data.mnist()

    # Turn the uint8 images into floating-point vectors.
    train_images = np.reshape(train_images,
                              (train_images.shape[0],
                               train_images.shape[1]*train_images.shape[2]))/255.0

    # Use one-hot coding for the labels.
    train_labels = kayak.util.onehot(train_labels)
    test_labels  = kayak.util.onehot(test_labels)

    # Hand the training data off to a cross-validation object.
    # This will create ten folds and allow us to easily iterate.
    CV = kayak.CrossValidator(num_folds, train_images, train_labels)

    valid_acc = 0.0

    # Loop over our cross validation folds.
    for ii, fold in enumerate(CV):
    
        # Get the training and validation data, according to this fold.
        train_images, train_labels = fold.train()
        valid_images, valid_labels = fold.valid()

        # Train on these data and get a prediction function back.
        pred_func = train(train_images, train_labels,
                          batch_size,
                          10.0**log10_lr,
                          momentum,
                          10.0**log10_l1,
                          10.0**log10_l2,
                          dropout,
                          improvement_thresh)

        # Make predictions on the validation data.
        valid_preds = np.argmax(pred_func( valid_images ), axis=1)

        # How did we do?
        acc = np.mean(valid_preds == np.argmax(valid_labels, axis=1))
        print "Fold %02d: %0.6f" % (ii+1, acc)
        valid_acc += acc
    

    print "Overall: %0.6f" % (valid_acc / num_folds)
    return valid_acc / num_folds

def main():
    # Define parameters to optimize
    parameters = { 'batch_size' : {'type':'integer', 'min':1,  'max':1024, 'size':1},
                   'log10_lr'   : {'type':'float',   'min':-6, 'max':0,    'size':1},
                   'momentum'   : {'type':'float',   'min':0,  'max':1,    'size':1},
                   'log10_l1'   : {'type':'float',   'min':-6, 'max':6,    'size':1},
                   'log10_l2'   : {'type':'float',   'min':-6, 'max':6,    'size':1},
                   'dropout'    : {'type':'float',   'min':0,  'max':1,    'size':1},
                   'improvement_thresh': {'type':'integer', 'min':0, 'max':25, 'size':1}}

    name = 'MNIST: Logistic Regression'
    description = 'Use logistic regression on the MNIST data.'
    outcome = {'name':'Cross-validation accuracy', 'type':'float'}
    scientist = whetlab.Experiment(name=name,
                                   description=description,
                                   parameters=parameters,
                                   outcome=outcome)

    while True:

        # Get suggested new experiment
        job = scientist.suggest()

        # Perform experiment
        print job
        outcome = evaluate(**job)

        # Inform scientist about the outcome
        scientist.update(job,outcome)
        print "Reported value %f" % (outcome)

if __name__ == '__main__':
    sys.exit(main())

