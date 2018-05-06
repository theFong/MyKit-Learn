# Put this inside the homeworks folder and simply execute python3 grading.py

import numpy as np
import sys
import argparse
import os
from sklearn import datasets
from sklearn.model_selection import train_test_split
import sys


def load_digits():
    digits = datasets.load_digits()
    x = digits.data/16
    x = x.reshape([x.shape[0], -1])
    y = digits.target
    return train_test_split(x, y, random_state=42, test_size=0.25)


def compute_membership(n_cluster, x, means, variances, pi_k):
    def gaussian_pdf(mean, variance):
        D = variance.shape[0]
        flag = False
        while not flag:
            try:
                inv_var = np.linalg.inv(variance)
                flag = True
            except np.linalg.LinAlgError:
                variance = variance + np.eye(D) * 1e-3
                flag = False

        inv = np.linalg.inv(variance)
        c = 2*np.pi*np.linalg.det(variance)

        def p(x):
            return np.exp(-0.5 * (x-mean) @ inv @ (x-mean).T) / np.sqrt(c)
        return p

    # copied and modified from sol/gmm.py
    gaussians = [
        gaussian_pdf(means[i], variances[i]) for i in range(n_cluster)
    ]

    N, D = x.shape
    membership = np.zeros((N, n_cluster))
    for i in range(N):
        for j in range(n_cluster):
            membership[i][j] = pi_k[j] * gaussians[j](x[i])
    return membership / np.sum(membership, axis=1).reshape([-1, 1])


def compute_log_likelihood(x, means, variances, pi_k):
    '''
        Return log-likelihood for the data

        x is a NXD matrix
        return : a float number which is the log-likelihood of data
    '''
    assert len(x.shape) == 2,  'x can only be 2 dimensional'

    def gaussian_pdf(mean, variance):
        D = variance.shape[0]
        flag = False
        while not flag:
            try:
                inv_var = np.linalg.inv(variance)
                flag = True
            except np.linalg.LinAlgError:
                variance = variance + np.eye(D) * 1e-3
                flag = False

        inv = np.linalg.inv(variance)
        c = ((2*np.pi)**D)*np.linalg.det(variance)

        def p(x):
            return np.exp(-0.5 * (x-mean) @ inv @ (x-mean).T) / np.sqrt(c)
        return p
    n_cluster = means.shape[0]

    gaussians = [gaussian_pdf(means[i], variances[i]) for i in range(n_cluster)]

    N, D = x.shape
    L = 0
    for i in range(N):
        p = 0
        for j in range(n_cluster):
            p = p + pi_k[j]*gaussians[j](x[i])
        if (p == 0):
            L = L-100000
        else:
            L = L + np.log(p)
    return float(L)


def transform_image(image, code_vectors):
    # copied from sol/kmeansTest.py
    '''
        Quantize image using the code_vectors

        Return new image from the image by replacing each RGB value in image with nearest code vectors (nearest in euclidean distance sense)

        returns:
            numpy array of shape image.shape
    '''

    assert image.shape[2] == 3 and len(image.shape) == 3, \
        'Image should be a 3-D array with size (?,?,3)'

    assert code_vectors.shape[1] == 3 and len(code_vectors.shape) == 2, \
        'code_vectors should be a 2-D array with size (?,3)'

    # TODO
    # - comment/remove the exception
    # - implement the function

    # DONOT CHANGE CODE ABOVE THIS LINE
    im = image
    N, M = im.shape[:2]
    n_codes = code_vectors.shape[0]
    distances = np.zeros((N, M, n_codes))
    for i in range(n_codes):
        distances[:, :, i] = np.sum((im - code_vectors[i])**2, axis=2)
    idx = np.argmin(distances, axis=2)
    new_im = code_vectors[idx]
    return new_im
    # DONOT CHANGE CODE BELOW THIS LINE


class KMeansClassifier():
    # copied and modified from sol/kmeans.py
    '''
        Class KMeansClassifier:
        Attr:
            n_cluster - Number of cluster for kmeans clustering
            max_iter - maximum updates for kmeans clustering
            e - error tolerance
    '''

    def __init__(self, n_cluster, max_iter=100, e=1e-6):
        self.n_cluster = n_cluster
        self.max_iter = max_iter
        self.e = e

    def fit(self, x, y, membership, centroids):
        '''
            Train the classifier
            params:
                x - N X D size  numpy array
                y - N size numpy array of labels
            returns:
                None
            Stores following attributes:
                self.centroids : centroids obtained by kmeans clustering
                self.centroid_labels : labels of each centroid obtained by
                    majority voting
        '''

        assert len(x.shape) == 2, "x should be a 2-D numpy array"
        assert len(y.shape) == 1, "y should be a 1-D numpy array"
        assert y.shape[0] == x.shape[0], "y and x should have same rows"

        np.random.seed(42)
        N, D = x.shape
        # TODO:
        # - comment/remove the exception.
        # - Implement the classifier
        # - assign means to centroids
        # - assign labels to centroid_labels

        # DONOT CHANGE CODE ABOVE THIS LINE
        # raise Exception(
        #     'Implement fit function in KMeansClassifier class (filename: kmeans.py')

        centroid_labels = []
        for i in range(self.n_cluster):
            y_ = y[(membership == i)]
            if (len(y_) == 0):
                centroid_labels.append(0)
            else:
                _, idx, counts = np.unique(
                    y_, return_index=True, return_counts=True)
                index = idx[np.argmax(counts)]
                centroid_labels.append(y_[index])
        centroid_labels = np.array(centroid_labels)
        # DONOT CHANGE CODE BELOW THIS LINE

        self.centroid_labels = centroid_labels
        self.centroids = centroids

        assert self.centroid_labels.shape == (
            self.n_cluster,
        ), 'centroid_labels should be a vector of shape {}'.format(
            self.n_cluster)

        assert self.centroids.shape == (
            self.n_cluster,
            D), 'centroid should be a numpy array of shape {} X {}'.format(
                self.n_cluster, D)

    def predict(self, x):
        '''
            Predict function

            params:
                x - N X D size  numpy array
            returns:
                predicted labels - numpy array of size (N,)
        '''

        assert len(x.shape) == 2, "x should be a 2-D numpy array"

        np.random.seed(42)
        N, D = x.shape
        # TODO:
        # - comment/remove the exception.
        # - Implement the prediction algorithm
        # - return labels

        # DONOT CHANGE CODE ABOVE THIS LINE
        # raise Exception(
        #     'Implement predict function in KMeansClassifier class (filename: kmeans.py')
        n_cluster = self.centroids.shape[0]
        N = x.shape[0]
        distances = np.zeros((N, n_cluster))

        for i in range(n_cluster):
            distances[:, i] = np.sum((x - self.centroids[i])**2, axis=1)
        centroid_idx = np.argmin(distances, axis=1)
        labels = []
        for i in centroid_idx:
            labels.append(self.centroid_labels[i])
        return np.array(labels)
        # DONOT CHANGE CODE BELOW THIS LINE


def test_kmeans_toy(results):
    print("K-Means Toy:")
    '''
    MAX SCORE : 2
        - Test convergence in less than 10 iters [0.5]
        - Test correct labelling >180 should be correct [0.5]

        - Ideal solution takes 2 iteration to converge
        - Ideal solution labels 197 correctly
    '''
    score = 0
    try:
        f_name = 'k_means_toy.npz'
        data = np.load('{}/{}'.format(results, f_name))
        centroids, step, membership, y = data['centroids'], data['step'], data[
            'membership'], data['y']

        # compute permutation y:membership
        perm, correct = {}, 0
        for i in range(4):
            temp = membership[y == i]
            unique, counts = np.unique(temp, return_counts=True)
            perm[i] = unique[np.argmax(counts)]
            correct += np.max(counts)

        # check if perm has all unique value
        unique_perm = len(np.unique(list(perm.values()))) == 4
        k = 190
        if (unique_perm and correct > k):
            score += 0.5
            print("\tLabelling score : 0.5/0.5")
        elif (unique_perm and correct > k*0.8):
            score += 0.25
            print("\tLabelling score : 0.25/0.5")
        elif (unique_perm and correct > k*0.6):
            score += 0.1
            print("\tLabelling score : 0.1/0.5")

        if step < 10:
            score += 0.5
            print('\tconvergence steps score : (0.5) ')
        elif step < 20:
            score += 0.25
            print('\tconvergence steps score : (0.25) ')
        elif step < 30:
            score += 0.1
            print('\tconvergence steps score : (0.1) ')
    except e:
        print(e)
    finally:
        print("K-means (toy) score (2x) :{} ".format(score))
        return score * 2, 1 * 2


def test_kmeans_compression(results):
    print("K-means compression:")
    '''
    MAX SCORE : 2
        - Check if centroid computed are ok and computed in less than 70 steps, i.e error after computing transform is <0.011
        - Check if computed reported error is <0.011

        - Ideal solution has 0.00973 pixel error, 36 steps
    '''
    score = 0
    try:
        f_name = 'k_means_compression.npz'
        data = np.load('{}/{}'.format(results, f_name))
        centroids, step, new_im, pixel_error, im = data['centroids'], data[
            'step'], data['new_image'], data['pixel_error'], data['im']

        N, M = im.shape[:2]
        transformed_image = transform_image(im, centroids)
        error = np.sum((transformed_image - im)**2) / (N * M)

        E = 0.01

        if (error < E and step < 70):
            score += 0.5
            print('\tComputed pixel rmse {}, score : {}'.format(error, 0.5))
        elif (error < 2*E and step < 70):
            score += 0.25
            print('\tComputed pixel rmse {}, score : {}'.format(error, 0.25))
        elif (error < 4*E and step < 100):
            score += 0.1
            print('\tComputed pixel rmse {}, score : {}'.format(error, 0.1))

        if pixel_error < E:
            score += 0.5
            print('\tReported pixel rmse {}, score : {}'.format(error, 0.5))
        elif pixel_error < 2*E:
            score += 0.25
            print('\tReported pixel rmse {}, score : {}'.format(error, 0.25))
        elif pixel_error < 4*E:
            score += 0.1
            print('\tReported pixel rmse {}, score : {}'.format(error, 0.1))
    except Exception as e:
        print(e)
    finally:
        print("K-means (compression) score (2x):{} ".format(score))
        return score * 2, 1 * 2


def test_kmeans_classification(results):
    print('K-Means classification:')
    '''
        MAX score = 2
        - Check the accuracy of k-means classifier is > 0.9

        - Ideal solution has accuracy 0.917
        NOTE: for now we just check accuracy but if needed we can do more checks we have centroids and centroid labels in results too
    '''

    score = 0
    try:
        f_name = 'k_means_classification.npz'
        data = np.load('{}/{}'.format(results, f_name))
        y_hat_test, y_test, centroids, centroid_labels = data[
            'y_hat_test'], data['y_test'], data['centroids'], data[
                'centroid_labels']

        acc = np.mean(y_hat_test == y_test)
        if acc > 0.9:
            score += 1.0
        elif acc > 0.8:
            score += 0.5
        elif acc > 0.7:
            score += 0.1
    except e:
        print(e)
    finally:
        print("K-Means classification score (2x) : {}".format(score))
        return score * 2, 1 * 2


def __test_gmm_results(
        data, n_cluster, max_updates, min_LL,  string):

    iterations, variances, pi_k, means, log_likelihood, x, y = data[
        'iterations'], data['variances'], data['pi_k'], data['means'], data[
            'log_likelihood'], data['x'], data['y']

    score = 0

    # check number of iterations
    if (iterations < max_updates):
        score += 0.25
        print('\tRequired iterations : {}/{}, score: 0.25'.format(iterations, max_updates))
    else:
        print('\tRequired iterations : {}/{}, score: 0'.format(iterations, max_updates))

    # check likelihood matches the actual value
    if (np.abs(log_likelihood-min_LL) < 0.1*np.abs(min_LL)):
        score += 0.5
        print('\tReported Likelihood : {}, score: 0.5'.format(log_likelihood))
    elif(np.abs(log_likelihood-min_LL) < 0.2*np.abs(min_LL)):
        score += 0.5/2
        print('\tReported Likelihood : {}, score: 0.25'.format(log_likelihood))
    else:
        print('\tReported Likelihood : {}, score: 0'.format(log_likelihood))

    # Check likelihood function is correctly implemented
    if (string == 'digits'):
        x, _, _, _ = load_digits()
    ll = compute_log_likelihood(x, means, variances, pi_k)
    if (np.abs(ll-log_likelihood) / np.abs(ll) < 0.05):
        score += 0.25
        print('\tcomputed likelihood : {}, score: 0.25'.format(log_likelihood))
    else:
        print('\tcomputed likelihood : {}, score: 0'.format(log_likelihood))
    print('\tTotal score {}'.format(score))
    return score, 1


def test_gmm_toy(results):
    ''' Test GMM results on toy dataset, random init + kmeans init
        - Check on number of iterations (0.25) + log_likelihood_value (0.5)
        - log_ll_fn (0.5)

        - actual iterations : 29 (random), 9 (k-means)
        - LL : -1663.269 (both, approx)
    '''
    score, max_score = 0, 0

    params = [{
        'fname': 'gmm_toy_random.npz',
        'max_updates': 50
    }, {
        'fname': 'gmm_toy_k_means.npz',
        'max_updates': 20
    }]
    for param in params:
        try:
            print('GMM {}:'.format(param['fname']))
            data = np.load('{}/{}'.format(results, param['fname']))
            s, _, = __test_gmm_results(
                data, 4, param['max_updates'], -1700, 'toy')
            score += s
        except Exception as e:
            print(e)
        finally:
            max_score += 1

    return score, max_score


def test_gmm_digits(results):
    ''' Test GMM results on digits dataset, random init + kmeans init
        - Check on number of iterations (0.25) + log_likelihood (0.5)
        - ll function (0.25)

        - actual iterations : 10 (random), 14 (k-means)
        - LL : 126125.974 (k-means), 120308.442 (random)
    '''

    params = [{
        'fname': 'gmm_digits_random.npz',
        'max_updates': 50,
    }, {
        'fname': 'gmm_digits_k_means.npz',
        'max_updates': 50,
    }]

    score, max_score = 0, 0
    result_detailed = {}
    for param in params:
        try:
            print('GMM {}:'.format(param['fname']))
            data = np.load('{}/{}'.format(results, param['fname']))
            s, _ = __test_gmm_results(data, 30, param['max_updates'], 119000,
                                      'digits')
            score += s
        except Exception as e:
            print(e)
        finally:
            max_score += 1
    return score, max_score

# check basic sampling (0.5)


def test_sampling(path):
    import importlib
    sys.path.append(path)
    try:
        spec = importlib.util.spec_from_file_location('GMM', path+'/gmm.py')
        gmm_mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(gmm_mod)
        gmm = gmm_mod.GMM(2)
        # means could be random?
        gmm.means = np.array([[1, -1], [-1, 1]])
        gmm.pi_k = np.array([0.3, 0.7])

        mean_of_means = ((gmm.means.T*gmm.pi_k).T).sum(axis=0)

        gmm.variances = np.array([np.eye(2), np.eye(2)])

        x = gmm.sample(10000)
        # print("Error:"+str(np.linalg.norm(np.mean(x, axis=0)-mean_of_means)))
        if (x.shape == (10000, 2) and np.linalg.norm(np.mean(x, axis=0)-mean_of_means) <= 0.02):
            print("Sampling error: {}, score 0.5 ".format(
                np.linalg.norm(np.mean(x, axis=0)-mean_of_means)))
            return 0.5, 0.5
        print("Sampling error: {}, score 0 ".format(
            np.linalg.norm(np.mean(x, axis=0)-mean_of_means)))
        return 0, 0.5
    except Exception as e:
        return 0, 0.5


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Grading script for PA4')
    parser.add_argument('-r', help="assignment folder", default="./")
    args = parser.parse_args()

    score, max_score = 0, 0
    s1, m1 = test_kmeans_toy(args.r+'/results/')
    score += s1
    max_score += m1

    s2, m2 = test_kmeans_compression(args.r+'/results/')
    score += s2
    max_score += m2

    s3, m3 = test_kmeans_classification(args.r+'/results/')

    score += s3
    max_score += m3

    s4, m4 = test_gmm_toy(args.r+'/results/')

    score += s4
    max_score += m4

    s5, m5 = test_gmm_digits(args.r+'/results/')
    score += s5
    max_score += m5

    s6, m6 = test_sampling(args.r)
    score += s6
    max_score += m6
    print(s1, m1, s2, m2, s3, m3, s4, m4, s5, m5, s6, m6, score, max_score)
    fout = open("output.csv", 'w')
    fout.write(str(s1) + "," + str(m1) + "\n")
    fout.write(str(s2) + "," + str(m2) + "\n")
    fout.write(str(s3) + "," + str(m3) + "\n")
    fout.write(str(s4) + "," + str(m4) + "\n")
    fout.write(str(s5) + "," + str(m5) + "\n")
    fout.close()
