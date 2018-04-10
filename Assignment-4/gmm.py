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
            self.pi_k  = np.zeros((self.n_cluster,D))

            # for xn, mem in zip(x,k_means_memberships):
            #     self.pi_k[mem] += xn
            # self.pi_k /= N

            # self.variances = np.zeros((self.n_cluster,D,D))
            # for xn, mem in zip(x,k_means_memberships):
            #     self.variances[mem] += ((xn - self.means[mem]) @ (xn - self.means[mem]).T)

            # for k in range(self.n_cluster):
            #     self.variances[k] /= (self.pi_k[k] * N)


            votes = np.bincount(memberships)
            self.variances = []
            self.pi_k = []
            for i in range(0, self.n_cluster):
                mu = self.means[i]
                num_members = votes[i]
                member_inds = np.where(memberships == i)[0]
                members = np.array([x[i] for i in member_inds])
                members = np.subtract(members, mu)
                variance = (members.T @ members)/num_members
                self.variances.append(variance)
                self.pi_k.append(num_members/N)

        elif (self.init == 'random'):
            # - initialize means randomly
            # - compute variance and pi_k

            self.means = np.random.uniform(0,1,(self.n_cluster,D))
            self.variances = np.array(np.identity((D,D)).tolist() * self.n_cluster)
            self.pi_k = np.array([[1. / self.n_cluster] * D] * self.n_cluster)


        else:
            raise Exception('Invalid initialization provided')

        # TODO
        # - find the optimal means, variances, and pi_k and assign it to self
        # - return number of updates done to reach the optimal values.
        # Hint: Try to seperate E & M step for clarity
        # l = self.compute_log_likelihood(x)
        l = 0
        update_count = self.max_iter
        for iter in range(self.max_iter):
            l_new = 0
            # gamma_memo = np.zeros((N,self.n_cluster))
            pi_k = np.zeros((self.n_cluster))
            means = np.zeros((self.n_cluster, D))
            variances = np.zeros((self.n_cluster,D,D))
            for k in range(self.n_cluster):
                n_k = 0
                u_k_num = np.zeros(D)
                var_k_num = np.zeros((D,D))
                for xn in x:
                    # E step (expectation)
                    gamma_ik_num = self.pi_k[k] * self.multi_gaussian(xn, self.means[k], self.variances[k])
                    gamma_ik_den = 0
                    for k_p in range(self.n_cluster):
                        gamma_ik_den += self.pi_k[k_p] * self.multi_gaussian(xn, self.means[k_p], self.variances[k_p])
                    gamma_ik = gamma_ik_num / gamma_ik_den

                    # M step (maximization)
                    n_k += gamma_ik
                    u_k_num += gamma_ik * xn
                    diff = (xn - self.means[k])

                    var_k_num += gamma_ik * ( diff @ diff.T )

                    l_new += np.log( gamma_ik_num )

                pi_k[k] = n_k / N
                means[k] = u_k_num / n_k
                variances[k] = var_k_num / n_k
            print(pi_k)
            print(means)
            print(variances)
            self.pi_k = pi_k
            self.means = means
            self.variances = variances

            # l_new = self.compute_log_likelihood(x)
            if np.abs(l - l_new) <= self.e:
                update_count = iter
                break
            l = l_new

        # gammas = np.zeros((N,self.n_cluster))
        # for i,xi in enumerate(x):
        #     for k in range(self.n_cluster):
        #         gammas[i][k] = self.multi_gaussian(xi,self.means[k], self.variances[k])

        # pi_k = np.sum(gammas, axis=0) / N
        # gamma_denom = np.sum(pi_k )

                

        return update_count + 1
    
    # normal dist x
    # 1 / (2 * pi ^ D * det(var) ) ^ (1/2) * exp(-1/2 * (x - u)^T * inv(var) * (x - u))
    # @ = matmul
    def multi_gaussian(self, xi, uk, vark):
        # print(xi, uk, vark)
        while la.cond(vark) > 1 / sys.float_info.epsilon:
            vark += .001 * np.identity(vark.shape[0])
        # print( np.exp(-1/2 * ((xi - uk).T @ la.inv(vark) @ (xi - uk))) )
        # print( (np.power(2*np.pi, xi.shape[0]) * la.det(vark)) )
        pdf = (1 / np.sqrt( np.power(2*np.pi, xi.shape[0]) * la.det(vark)) ) * np.exp( -(1 / 2) * ((xi - uk).T @ la.inv(vark) @ (xi - uk)) )
        # return (1/np.sqrt(np.power(2*np.pi, x.shape[0]) * np.linalg.det(variance))) * np.exp(-(1/2) * (x-mean) * np.linalg.inv(variance) * (x-mean).T)
        # print('pdf: ' , pdf)
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
        # TODO
        # - comment/remove the exception
        # - calculate log-likelihood using means, variances and pi_k attr in self
        # - return the log-likelihood
        # Note: you can call this function in fit function (if required)
        # DONOT MODIFY CODE ABOVE THIS LINE
        raise Exception('Implement compute_log_likelihood function in gmm.py')
        # DONOT MODIFY CODE BELOW THIS LINE
