import numpy as np
import numpy.linalg as la

def pca(X = np.array([]), no_dims = 50):
    """
    Runs PCA on the N x D array X in order to reduce its dimensionality to 
     no_dims dimensions.
    Inputs:
    - X: A matrix with shape N x D where N is the number of examples and D is 
         the dimensionality of original data.
    - no_dims: A scalar indicates the output dimension of examples after 
         performing PCA.
    Returns:
    - Y: A matrix of reduced size with shape N x no_dims where N is the number
         of examples  and no_dims is the dimensionality of output examples. 
         no_dims should be smaller than D, which is the dimensionality of 
         original examples.
    - M: A matrix of eigenvectors with shape D x no_dims where D is the 
         dimensionality of the original data
    """
    Y = np.array([])
    M = np.array([])

    cov_mat = np.cov(X.T)
    eig_vals, eig_vecs = la.eigh(cov_mat)
    # l2 normalize of each eig vec
    eig_vec_norms = np.array([ la.norm(e) for e in eig_vecs.T ])
    eig_vecs /= eig_vec_norms
    # tuple of eigen values and their respective vector
    eig_pairs = [ (e_val, e_vec) for e_val, e_vec in zip(eig_vals, eig_vecs.T) ]
    # sort largest to smallest based on eigen values 
    eig_pairs.sort(key = lambda pair : pair[0], reverse = True)
    # select top no_dims
    M = np.array([ e[1] for e,_ in zip(eig_pairs, range(no_dims)) ]).T
    # transform
    Y = X @ M
    return Y, M

def decompress(Y = np.array([]), M = np.array([])):
    """
    Returns compressed data to initial shape, hence decompresses it.
    Inputs:
    - Y: A matrix of reduced size with shape N x no_dims where N is the number
         of examples  and no_dims is the dimensionality of output examples. 
         no_dims should be smaller than D, which is the dimensionality of 
         original examples.
    - M: A matrix of eigenvectors with shape D x no_dims where D is the 
         dimensionality of the original data
    Returns:
    - X_hat: Reconstructed matrix with shape N x D where N is the number of 
         examples and D is the dimensionality of each example before 
         compression.
    """
    X_hat = np.array([])

    X_hat = Y @ M.T
    
    return X_hat

def reconstruction_error(orig = np.array([]), decompressed = np.array([])):
    """
    Computes reconstruction error (pixel-wise mean squared error) for original
     image and reconstructed image
    Inputs:
    - orig: An array of size 1xD, original flattened image.
    - decompressed: An array of size 1xD, decompressed version of the image
    """

    error = ((orig - decompressed) ** 2).mean(axis=None)
    
    return error

def load_data(dataset='mnist_subset.json'):
    # This function reads the MNIST data
    import json


    with open(dataset, 'r') as f:
        data_set = json.load(f)
    mnist = np.vstack((np.asarray(data_set['train'][0]), 
                    np.asarray(data_set['valid'][0]), 
                    np.asarray(data_set['test'][0])))
    return mnist

if __name__ == '__main__':
    
    import argparse
    import sys


    mnist = load_data()
    compression_rates = [2, 10, 50, 100, 250, 500]
    with open('pca_output.txt', 'w') as f:
        for cr in compression_rates:
            Y, M = pca(mnist - np.mean(mnist, axis=0), cr)
            
            decompressed_mnist = decompress(Y, M)
            decompressed_mnist += np.mean(mnist, axis=0)
            
            total_error = 0.
            for mi, di in zip(mnist, decompressed_mnist):
                error = reconstruction_error(mi, di)
                f.write(str(error))
                f.write('\n')
                total_error += error
            print('Total reconstruction error after compression with %d principal '\
                'components is %f' % (cr, total_error))



