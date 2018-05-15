from __future__ import division, print_function

import numpy as np
import scipy as sp

from matplotlib import pyplot as plt
from matplotlib import cm

# supress sklearn warnings
def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

from tqdm import tqdm

def binary_train(X, y, w0=None, b0=None, step_size=0.5, max_iterations=1000):
    """
    Inputs:
    - X: training features, a N-by-D numpy array, where N is the 
    number of training points and D is the dimensionality of features
    - y: binary training labels, a N dimensional numpy array where 
    N is the number of training points, indicating the labels of 
    training data
    - step_size: step size (learning rate)

    Returns:
    - w: D-dimensional vector, a numpy array which is the weight 
    vector of logistic regression
    - b: scalar, which is the bias of logistic regression

    Find the optimal parameters w and b for inputs X and y.
    Use the average of the gradients for all training examples to
    update parameters.
    """
    
    N, D = X.shape
    assert len(np.unique(y)) == 2

    w = np.zeros(D)
    if w0 is not None:
        w = w0
    
    b = 0
    if b0 is not None:
        b = b0
    
    w = np.append(w, b)
    X = np.append(X, [[1]]*len(X), axis=1)
    D += 1

    """
    Stochastic gradient descent:
    w  <- w - step_size * (sigmoid(W^tXn) - Yn) * Xn
    """

    for it in tqdm(range(0,max_iterations), ncols=100):
        # calculate gradient
        g = np.mean([ ((sigmoid(w.T.dot(Xn)) - Yn) * Xn) for Xn, Yn in zip(X,y) ], axis=0)
        # update rule
        w -= step_size * g

    b = w[D-1]
    assert w.shape == (D,)
    return w, b


def binary_predict(X, w, b):
    """
    Inputs:
    - X: testing features, a N-by-D numpy array, where N is the 
    number of training points and D is the dimensionality of features
    
    Returns:
    - preds: N dimensional vector of binary predictions: {0, 1}
    """
    X = np.append(X, [[1]]*len(X), axis=1)
    N, D = X.shape
    preds = np.zeros(N) 
    w = np.array(w)
    

    """
    h(x) = sigmoid(W^tX)
    """    
    preds = sigmoid(np.inner(w.T,(X)))
    for i in range(len(preds)):
        preds[i] = 1 if preds[i] > .5 else 0

    assert preds.shape == (N,) 
    return preds


def multinomial_train(X, y, C, 
                     w0=None, 
                     b0=None, 
                     step_size=0.5, 
                     max_iterations=1000):
    """
    Inputs:
    - X: training features, a N-by-D numpy array, where N is the 
    number of training points and D is the dimensionality of features
    - y: multiclass training labels, a N dimensional numpy array where
    N is the number of training points, indicating the labels of 
    training data
    - C: number of classes in the data
    - step_size: step size (learning rate)
    - max_iterations: maximum number for iterations to perform

    Returns:
    - w: C-by-D weight matrix of multinomial logistic regression, where 
    C is the number of classes and D is the dimensionality of features.
    - b: bias vector of length C, where C is the number of classes

    Implement a multinomial logistic regression for multiclass 
    classification. Keep in mind, that for this task you may need a 
    special (one-hot) representation of classification labels, where 
    each label y_i is represented as a row of zeros with a single 1 in
    the column, that corresponds to the class y_i belongs to. 
    """
    N, D = X.shape

    w = np.zeros((C, D))
    if w0 is not None:
        w = w0
    
    b = np.zeros(C)
    if b0 is not None:
        b = b0

    w = np.append(w, [[b_] for b_ in b], axis=1)
    X = np.append(X, [[1]]*len(X), axis=1)
    D += 1

    """
    stochastic gradient descent:
    Gn = exp(Wn^tX) / sum(exp(Wi^tX))
    1 of k encoding
    w <- w - step_size * Gn
    """
    # 1 of k encoding
    one_of_k_memo = [[1 if i == k else 0 for k in range(0,C)] for i in range(0,C) ]
    y = np.array([ one_of_k_memo[Yn] for Yn in y])
    # sgd
    for it in tqdm(range(0,max_iterations), ncols= 100):
        # calculate the gradients
        # g = np.mean([ calc_gradient(w, Xn, Yn) for Xn, Yn in zip(X,y) ], axis=0)
        g = np.zeros((C,D+1))
        for Xn, Yn in zip(X,y):
            # update gradient at that point
            g = ( calc_gradient(w, Xn, Yn) / N )
            w = w - step_size * g

        # update rule
        
        # update 

    b = w[:,D-1]
    assert w.shape == (C, D)
    assert b.shape == (C,)
    return w, b

def calc_gradient(w, Xn, Yn):
    soft_max_nums = [ np.exp(Wl.T.dot(Xn)) for Wl in w ]
    soft_max_denom = np.sum(soft_max_nums)
    return np.array([ ((snk / soft_max_denom) - Yk) * Xn for Yk,snk in zip(Yn, soft_max_nums)])

def multinomial_predict(X, w, b):
    """
    Inputs:
    - X: testing features, a N-by-D numpy array, where N is the 
    number of training points and D is the dimensionality of features
    - w: weights of the trained multinomial classifier
    - b: bias terms of the trained multinomial classifier
    
    Returns:
    - preds: N dimensional vector of multiclass predictions.
    Outputted predictions should be from {0, C - 1}, where
    C is the number of classes

    Make predictions for multinomial classifier.
    """
    X = np.append(X, [[1]]*len(X), axis=1)
    N, D = X.shape
    C = w.shape[0]
    preds = np.zeros(N) 

    """
    argmax(W^tx)
    """

    softmax = np.inner(w,(X))
    preds = np.argmax(softmax,axis = 0)

    assert preds.shape == (N,)
    return preds


def OVR_train(X, y, C, w0=None, b0=None, step_size=0.5, max_iterations=150):
    """
    Inputs:
    - X: training features, a N-by-D numpy array, where N is the 
    number of training points and D is the dimensionality of features
    - y: multiclass training labels, a N dimensional numpy array, 
    indicating the labels of each training point
    - C: number of classes in the data
    - w0: initial value of weight matrix
    - b0: initial value of bias term
    - step_size: step size (learning rate)
    - max_iterations: maximum number of iterations for gradient descent

    Returns:
    - w: a C-by-D weight matrix of OVR logistic regression
    - b: bias vector of length C

    Implement multiclass classification using binary classifier and 
    one-versus-rest strategy. Recall, that the OVR classifier is 
    trained by training C different classifiers. 
    """
    N, D = X.shape
    w = np.zeros((C, D))
    if w0 is not None:
        w = w0
    
    b = np.zeros(C)
    if b0 is not None:
        b = b0

    """
    create train C logistic classifiers
    1 vs the rest
    """
    w = np.array([ train_bin_clasifier(X, y, w0, b0, i, step_size, max_iterations) for i in range(0, C) ])

    D += 1
    assert w.shape == (C, D), 'wrong shape of weights matrix'
    assert b.shape == (C,), 'wrong shape of bias terms vector'
    return w, b

def train_bin_clasifier(X, y, w, b, k, step_size, max_iterations):
    y = [ 1 if Yn == k else 0 for Yn in y ]
    return binary_train(X, y, w, b, step_size=step_size, max_iterations=max_iterations)[0]

def OVR_predict(X, w, b):
    """
    Inputs:
    - X: testing features, a N-by-D numpy array, where N is the 
    number of training points and D is the dimensionality of features
    - w: weights of the trained OVR model
    - b: bias terms of the trained OVR model
    
    Returns:
    - preds: vector of class label predictions.
    Outputted predictions should be from {0, C - 1}, where
    C is the number of classes.

    Make predictions using OVR strategy and predictions from binary
    classifier. 
    """
    N, D = X.shape
    C = w.shape[0]
    preds = np.zeros(N) 
    X = np.append(X, [[1]]*len(X), axis=1)
    """
    1 or the rest
    create K predictions and return max class
    """
    preds = [ sigmoid(np.inner(Wk.T,(X))) for Wk in w ]
    preds = np.argmax(preds, axis=0)

    assert preds.shape == (N,)
    return preds


#######################################################################
# DO NOT MODIFY THE CODE BELOW 
#######################################################################

def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def accuracy_score(true, preds):
    return np.sum(true == preds).astype(float) / len(true)

def run_binary():
    from data_loader import toy_data_binary, \
                            data_loader_mnist 

    print('Performing binary classification on synthetic data')
    X_train, X_test, y_train, y_test = toy_data_binary()
        
    w, b = binary_train(X_train, y_train)
    
    train_preds = binary_predict(X_train, w, b)
    preds = binary_predict(X_test, w, b)
    print('train acc: %f, test acc: %f' % 
            (accuracy_score(y_train, train_preds),
             accuracy_score(y_test, preds)))
    
    print('Performing binary classification on binarized MNIST')
    X_train, X_test, y_train, y_test = data_loader_mnist()

    binarized_y_train = [0 if yi < 5 else 1 for yi in y_train] 
    binarized_y_test = [0 if yi < 5 else 1 for yi in y_test] 
    
    w, b = binary_train(X_train, binarized_y_train)
    
    train_preds = binary_predict(X_train, w, b)
    preds = binary_predict(X_test, w, b)
    print('train acc: %f, test acc: %f' % 
            (accuracy_score(binarized_y_train, train_preds),
             accuracy_score(binarized_y_test, preds)))

def run_multiclass():
    from data_loader import toy_data_multiclass_3_classes_non_separable, \
                            toy_data_multiclass_5_classes, \
                            data_loader_mnist 
    
    datasets = [(toy_data_multiclass_3_classes_non_separable(), 
                        'Synthetic data', 3), 
                (toy_data_multiclass_5_classes(), 'Synthetic data', 5), 
                (data_loader_mnist(), 'MNIST', 10)]

    # datasets = [(toy_data_multiclass_3_classes_non_separable(), 
    #                     'Synthetic data', 3)]

    for data, name, num_classes in datasets:
        print('%s: %d class classification' % (name, num_classes))
        X_train, X_test, y_train, y_test = data
        
        # print('One-versus-rest:')
        # w, b = OVR_train(X_train, y_train, C=num_classes)
        # train_preds = OVR_predict(X_train, w=w, b=b)
        # preds = OVR_predict(X_test, w=w, b=b)
        # print('train acc: %f, test acc: %f' % 
        #     (accuracy_score(y_train, train_preds),
        #      accuracy_score(y_test, preds)))
    
        print('Multinomial:')
        w, b = multinomial_train(X_train, y_train, C=num_classes)
        train_preds = multinomial_predict(X_train, w=w, b=b)
        preds = multinomial_predict(X_test, w=w, b=b)
        print('train acc: %f, test acc: %f' % 
            (accuracy_score(y_train, train_preds),
             accuracy_score(y_test, preds)))


if __name__ == '__main__':
    
    import argparse
    import sys

    parser = argparse.ArgumentParser()
    parser.add_argument("--type", )
    parser.add_argument("--output")
    args = parser.parse_args()

    if args.output:
        sys.stdout = open(args.output, 'w')

    if not args.type or args.type == 'binary':
        run_binary()

    if not args.type or args.type == 'multiclass':
        run_multiclass()
