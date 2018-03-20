import json
import numpy as np
from numpy import linalg as la


###### Q1.1 ######
def objective_function(X, y, w, lamb):
    """
    Inputs:
    - Xtrain: A 2 dimensional numpy array of data (number of samples x number of features)
    - ytrain: A 1 dimensional numpy array of labels (length = number of samples )
    - w: a numpy array of D elements as a D-dimension vector, which is the weight vector and initialized to be all 0s
    - lamb: lambda used in pegasos algorithm

    Return:
    - train_obj: the value of objective function in SVM primal formulation
    """
    # ( lamb / 2 |W|l2^2  + 1 / n * sum n(max(0,1 - Yn * W^T Xn)))
    obj_value = lamb / 2 * la.norm(w)**2 + np.mean([max(0, 1 - Yn * np.matmul(w.T, Xn.reshape(Xn.shape[0],1))) for Xn, Yn in zip(X,y)])
    return obj_value.tolist()


###### Q1.2 ######
def pegasos_train(Xtrain, ytrain, w, lamb, k, max_iterations):
    """
    Inputs:
    - Xtrain: A list of num_train elements, where each element is a list of D-dimensional features.
    - ytrain: A list of num_train labels
    - w: a numpy array of D elements as a D-dimension vector, which is the weight vector and initialized to be all 0s
    - lamb: lambda used in pegasos algorithm
    - k: mini-batch size
    - max_iterations: the maximum number of iterations to update parameters

    Returns:
    - learnt w
    - traiin_obj: a list of the objective function value at each iteration during the training process, length of 500.
    """
    np.random.seed(0)
    Xtrain = np.array(Xtrain)
    ytrain = np.array(ytrain)
    N = Xtrain.shape[0]
    D = Xtrain.shape[1]

    train_obj = []

    for iter in range(1, max_iterations + 1):
        # randomly create k rand nums scale by N floor of values and cast as int
        A_t = np.floor(np.random.rand(k) * N).astype(int)  # indexes of the current mini-batch

        A_t_pos = []
        for i in A_t:
            if ytrain[i] * np.inner(w.transpose(),Xtrain[i]) < 1:
                A_t_pos.append(i)

        learn_rate = 1 / (lamb * iter)

        some_sum = [0] * D
        for d in A_t_pos:
            some_sum += Xtrain[i] * ytrain[i]
        some_sum = [[i] for i in some_sum]

        w = (1 - learn_rate * lamb) * w + np.multiply((learn_rate / k), some_sum)
        w = np.multiply( min( 1.0, (1 / (lamb ** .5) ) / la.norm(w) ), w)

        train_obj.append(objective_function(Xtrain, ytrain, w, lamb))


    # lh = (1 - n_t * lamb) * w 
    #     rh = np.multiply((n_t / k), b_sum)
    #     w_ = lh + rh
    #     z = ((1 / np.sqrt(lamb)) / np.linalg.norm(w_))
    #     if(z < 1):
    #         w = np.multiply(z, w_)
    #     else:
    #         w = w_
    #     train_obj.append(objective_function(Xtrain, ytrain, w, lamb))

    return w, train_obj


###### Q1.3 ######
def pegasos_test(Xtest, ytest, w, t = 0.):
    """
    Inputs:
    - Xtest: A list of num_test elements, where each element is a list of D-dimensional features.
    - ytest: A list of num_test labels
    - w_l: a numpy array of D elements as a D-dimension vector, which is the weight vector of SVM classifier and learned by pegasos_train()
    - t: threshold, when you get the prediction from SVM classifier, it should be real number from -1 to 1. Make all prediction less than t to -1 and otherwise make to 1 (Binarize)

    Returns:
    - test_acc: testing accuracy.
    """
    Xtest = np.array(Xtest)
    ytest = np.array(ytest)

    N = Xtest.shape[0]

    preds = np.inner(w.transpose(),Xtest)

    preds = [1 if n > t else -1 for n in preds[0]]
    
    correct_count = 0

    for p, y in zip(preds, ytest):
        if p == y:
            correct_count += 1
    test_acc = correct_count / N

    return test_acc



def data_loader_mnist(dataset):

    with open(dataset, 'r') as f:
            data_set = json.load(f)
    train_set, valid_set, test_set = data_set['train'], data_set['valid'], data_set['test']

    Xtrain = train_set[0]
    ytrain = train_set[1]
    Xvalid = valid_set[0]
    yvalid = valid_set[1]
    Xtest = test_set[0]
    ytest = test_set[1]

    ## below we add 'one' to the feature of each sample, such that we include the bias term into parameter w
    Xtrain = np.hstack((np.ones((len(Xtrain), 1)), np.array(Xtrain))).tolist()
    Xvalid = np.hstack((np.ones((len(Xvalid), 1)), np.array(Xvalid))).tolist()
    Xtest = np.hstack((np.ones((len(Xtest), 1)), np.array(Xtest))).tolist()

    for i, v in enumerate(ytrain):
        if v < 5:
            ytrain[i] = -1.
        else:
            ytrain[i] = 1.
    for i, v in enumerate(ytest):
        if v < 5:
            ytest[i] = -1.
        else:
            ytest[i] = 1.

    return Xtrain, ytrain, Xvalid, yvalid, Xtest, ytest


def pegasos_mnist():

    test_acc = {}
    train_obj = {}

    Xtrain, ytrain, Xvalid, yvalid, Xtest, ytest = data_loader_mnist(dataset = 'mnist_subset.json')

    max_iterations = 500
    k = 100
    for lamb in (0.01, 0.1, 1):
        w = np.zeros((len(Xtrain[0]), 1))
        w_l, train_obj['k=' + str(k) + '_lambda=' + str(lamb)] = pegasos_train(Xtrain, ytrain, w, lamb, k, max_iterations)
        test_acc['k=' + str(k) + '_lambda=' + str(lamb)] = pegasos_test(Xtest, ytest, w_l)

    lamb = 0.1
    for k in (1, 10, 1000):
        w = np.zeros((len(Xtrain[0]), 1))
        w_l, train_obj['k=' + str(k) + '_lambda=' + str(lamb)] = pegasos_train(Xtrain, ytrain, w, lamb, k, max_iterations)
        test_acc['k=' + str(k) + '_lambda=' + str(lamb)] = pegasos_test(Xtest, ytest, w_l)

    return test_acc, train_obj


def main():
    test_acc, train_obj = pegasos_mnist() # results on mnist
    print('mnist test acc \n')
    for key, value in test_acc.items():
        print('%s: test acc = %.4f \n' % (key, value))

    with open('pegasos.json', 'w') as f_json:
        json.dump([test_acc, train_obj], f_json)


if __name__ == "__main__":
    main()
