import numpy as np
import numpy.linalg as la
from kmeans import KMeans
import sys


class GMM():
    '''
        Fits a Gausian Mixture model to the data.

        attrs:
            n_cluster : Number of mixtures
            e : error tolerance
            max_iter : maximum number of updates
            init : initialization of means and variance
                Can be 'random' or 'kmeans'
            means : means of gaussian mixtures
            variances : variance of gaussian mixtures
            pi_k : mixture probabilities of different component
    '''

    def __init__(self, n_cluster, init='k_means', max_iter=100, e=0.0001):
        self.n_cluster = n_cluster
        self.e = e
        self.max_iter = max_iter
        self.init = init
        self.means = None
        self.variances = None
        self.pi_k = None
        self.gammas = None

    def fit(self, x):
        '''
            Fits a GMM to x.

            x: is a NXD size numpy array
            updates:
                self.means
                self.variances
                self.pi_k
        '''
        assert len(x.shape) == 2, 'x can only be 2 dimensional'

        np.random.seed(42)
        N, D = x.shape

        if (self.init == 'k_means'):
            # - initialize means using k-means clustering
            # - compute variance and pi_k
            km = KMeans(self.n_cluster, max_iter=self.max_iter, e=self.e)
            self.means, memberships, _  = km.fit(x)
            
            quorem = np.bincount(memberships)
            self.variances = np.zeros((self.n_cluster, D, D))
            self.pi_k = np.zeros((self.n_cluster))
            for k in range(self.n_cluster):
                num_members = quorem[k]
                member_inds = np.where(memberships == k)[0]
                members = np.array([x[i] for i in member_inds])
                mem_diff = members - self.means[k]
                variance = (mem_diff.T @ mem_diff)/num_members
                self.variances[k] = variance
                self.pi_k[k] = (num_members/N)


        elif (self.init == 'random'):
            # - initialize means randomly
            # - compute variance and pi_k
            self.means = np.random.uniform(0,1,(self.n_cluster,D))
            self.variances = np.array(([np.identity(D)] * self.n_cluster))
            self.pi_k = np.array([1. / self.n_cluster] * self.n_cluster)


        else:
            raise Exception('Invalid initialization provided')

        update_count = self.max_iter
        l = 0
        for iter in range(self.max_iter):
            # E step
            gammas = np.zeros((N,self.n_cluster))
            for i,xi in enumerate(x):
                for k in range(self.n_cluster):
                    gammas[i][k] = self.pi_k[k] * self.multi_gaussian(xi,self.means[k], self.variances[k])
            likelihood = np.sum(gammas, axis=1).reshape(N, 1)
            l_new = np.sum(gammas.T @ np.log(likelihood), axis= 0)[0]
            self.gammas = gammas / likelihood

            # M step
            n_k = np.sum(gammas, axis=0).reshape(self.n_cluster,1)
            self.pi_k = n_k / N
            self.means = (gammas.T @ x) / n_k

            self.variances = np.zeros((self.n_cluster, D, D))
            for k in range(self.n_cluster):
                diff = np.sqrt(gammas[:,k].reshape(N, 1)) * (x - self.means[k])
                self.variances[k] =  (diff.T @ diff) / n_k[k]
            
            # print(l_new)
            if np.abs(l - l_new) <= self.e:
                update_count = iter
                break
            l = l_new
                
        self.pi_k = self.pi_k.reshape(self.n_cluster,)
        return update_count + 1
    
    # normal dist x
    # 1 / (2 * pi ^ D * det(var) ) ^ (1/2) * exp(-1/2 * (x - u)^T * inv(var) * (x - u))
    # @ = matmul
    def multi_gaussian(self, xi, uk, vark):
        while la.cond(vark) > 1 / sys.float_info.epsilon:
            vark += .001 * np.identity(vark.shape[0])
        pdf = (1 / np.sqrt( np.power(2*np.pi, xi.shape[0]) * la.det(vark)) ) * np.exp( -(1 / 2) * ((xi - uk).T @ la.inv(vark) @ (xi - uk)) )
        return pdf

    def sample(self, N):
        '''
        sample from the GMM model

        N is a positive integer
        return : NXD array of samples

        '''
        assert type(N) == int and N > 0, 'N should be a positive integer'
        np.random.seed(42)
        if (self.means is None):
            raise Exception('Train GMM before sampling')

        # TODO
        # - comment/remove the exception
        # - generate samples from the GMM
        # - return the samples

        # DONOT MODIFY CODE ABOVE THIS LINE
        raise Exception('Implement sample function in gmm.py')
        # DONOT MODIFY CODE BELOW THIS LINE

    def compute_log_likelihood(self, x):
        '''
            Return log-likelihood for the data

            x is a NXD matrix
            return : a float number which is the log-likelihood of data
        '''
        assert len(x.shape) == 2,  'x can only be 2 dimensional'
        # - calculate log-likelihood using means, variances and pi_k attr in self
        # - return the log-likelihood
        gammas = np.zeros((len(x),self.n_cluster))
        for i,xi in enumerate(x):
            for k in range(self.n_cluster):
                gammas[i][k] = self.pi_k[k] * self.multi_gaussian(xi,self.means[k], self.variances[k])
        likelihood = np.sum(gammas, axis=1).reshape(len(x), 1)
        gammas = gammas / likelihood
        ll = np.sum(gammas.T @ np.log(likelihood), axis= 0)[0]
        # print(ll)
        # print(type(ll))
        return float(ll)

