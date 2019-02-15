from math import gamma

import numpy as np
from kmeans import KMeans


class GMM():
    '''
        Fits a Gausian Mixture model to the data.

        attrs:
            n_cluster : Number of mixtures (Int)
            e : error tolerance (Float) 
            max_iter : maximum number of updates (Int)
            init : initialization of means and variance
                Can be 'random' or 'kmeans' 
            means : means of Gaussian mixtures (n_cluster X D numpy array)
            variances : variance of Gaussian mixtures (n_cluster X D X D numpy array) 
            pi_k : mixture probabilities of different component ((n_cluster,) size numpy array)
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
            # TODO
            # - comment/remove the exception
            # - initialize means using k-means clustering
            # - compute variance and pi_k (see P4.pdf)

            # DONOT MODIFY CODE ABOVE THIS LINE
            kmeans = KMeans(n_cluster=self.n_cluster, max_iter=self.max_iter, e=self.e)
            self.means, membership, updates = kmeans.fit(x)
            gamma = np.identity(self.n_cluster)[membership]
            numberByCluster = gamma.sum(axis=0)
            self.pi_k = numberByCluster / N
            self.variances = np.zeros((self.n_cluster, D, D))
            for i in range(self.n_cluster):
                variance = x - self.means[i]
                self.variances[i] = np.dot(np.transpose(variance) * gamma[:,i],(variance)) / numberByCluster[i]
                #self.variances[i] = 1 / numberByCluster[i] * (((variance).transpose() * gamma[:, i]).dot(variance))
            # DONOT MODIFY CODE BELOW THIS LINE

        elif (self.init == 'random'):
            # TODO
            # - comment/remove the exception
            # - initialize means randomly
            # - initialize variance to be identity and pi_k to be uniform

            # DONOT MODIFY CODE ABOVE THIS LINE
            self.variances = np.zeros((self.n_cluster, D, D))
            self.means = np.random.uniform(0, 1, (self.n_cluster, D))
            #self.variances = np.array([np.eye(D)])
            for i in range(self.n_cluster):
                self.variances[i] = np.identity(D)
            self.pi_k = np.full((self.n_cluster,), 1. / self.n_cluster)
            gamma = np.empty((N, self.n_cluster))
            # DONOT MODIFY CODE BELOW THIS LINE

        else:
            raise Exception('Invalid initialization provided')

        # TODO
        # - comment/remove the exception
        # - Use EM to learn the means, variances, and pi_k and assign them to self
        # - Update until convergence or until you have made self.max_iter updates.
        # - Return the number of E/M-Steps executed (Int) 
        # Hint: Try to separate E & M step for clarity
        # DONOT MODIFY CODE ABOVE THIS LINE
        likelhood = self.compute_log_likelihood(x, means=self.means, variances=self.variances, pi_k=self.pi_k)
        for iter in range(self.max_iter):
            for k in range(self.n_cluster):
                gamma[:, k] = GMM.Gaussian_pdf(self.means[k], self.variances[k]).getLikelihood(x)

            gamma *= self.pi_k
            gamma /= gamma.sum(axis=1, keepdims=True)
            numberByCluster = gamma.sum(axis=0)
            self.means = np.transpose(gamma).dot(x) / np.transpose(numberByCluster[None, :])

            variance = x - self.means[:, None]
            for i in range(self.n_cluster):
                self.variances[i, ...] = (np.transpose(gamma[:, i,None] * variance[i]).dot(variance[i])) / numberByCluster[i]

            self.pi_k = numberByCluster/N
            likelhood_new = self.compute_log_likelihood(x, means=self.means, variances=self.variances, pi_k=self.pi_k)
            if np.abs(likelhood-likelhood_new) <= self.e:
                break
            likelhood = likelhood_new

        return iter

        # DONOT MODIFY CODE BELOW THIS LINE

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
        samples = []
        for k in np.random.choice(self.n_cluster, size=N, p=self.pi_k):
            samples.append(np.random.multivariate_normal(
                mean=self.means[k], cov=self.variances[k]
            ))
        return np.array(samples)
        # DONOT MODIFY CODE BELOW THIS LINE

    def compute_log_likelihood(self, x, means=None, variances=None, pi_k=None):
        '''
            Return log-likelihood for the data

            x is a NXD matrix
            return : a float number which is the log-likelihood of data
        '''
        assert len(x.shape) == 2, 'x can only be 2 dimensional'
        if means is None:
            means = self.means
        if variances is None:
            variances = self.variances
        if pi_k is None:
            pi_k = self.pi_k
            # TODO
        # - comment/remove the exception
        # - calculate log-likelihood using means, variances and pi_k attr in self
        # - return the log-likelihood (Float)
        # Note: you can call this function in fit function (if required)
        # DONOT MODIFY CODE ABOVE THIS LINE
        N, D = np.shape(x)
        joint_log_likelihood = np.zeros((N, self.n_cluster))

        for k in range(self.n_cluster):
            joint_log_likelihood[:, k] = GMM.Gaussian_pdf(means[k], variances[k]).getLikelihood(x)

        joint_log_likelihood *= self.pi_k
        log_likelihood = float(np.sum(np.log(np.sum(joint_log_likelihood, axis=1))))

        # raise Exception('Implement compute_log_likelihood function in gmm.py')
        # DONOT MODIFY CODE BELOW THIS LINE
        return log_likelihood

    class Gaussian_pdf():
        def __init__(self, mean, variance):
            self.mean = mean
            self.variance = variance
            self.c = None
            self.inv = None
            '''
                Input: 
                    Means: A 1 X D numpy array of the Gaussian mean
                    Variance: A D X D numpy array of the Gaussian covariance matrix
                Output: 
                    None: 
            '''
            # TODO
            # - comment/remove the exception
            # - Set self.inv equal to the inverse the variance matrix (after ensuring it is full rank - see P4.pdf)
            # - Set self.c equal to ((2pi)^D) * det(variance) (after ensuring the variance matrix is full rank)
            # Note you can call this class in compute_log_likelihood and fit
            # DONOT MODIFY CODE ABOVE THIS LINE
            row, column = np.shape(variance)
            inversible_variance = self.get_inversable_matrix(row)
            self.inv = np.linalg.inv(inversible_variance)
            self.c = ((2 * np.pi) ** row) * np.linalg.det(inversible_variance)
            # raise Exception('Impliment Guassian_pdf __init__')
            # DONOT MODIFY CODE BELOW THIS LINE

        def get_inversable_matrix(self, dimension):
            copy_v = np.copy(self.variance)
            while np.linalg.matrix_rank(copy_v) < dimension:
                copy_v += np.identity(dimension) * (1e-3)
            return copy_v

        def getLikelihood(self, x):
            '''
                Input: 
                    x: a 1 X D numpy array representing a sample
                Output: 
                    p: a numpy float, the likelihood sample x was generated by this Gaussian
                Hint: 
                    p = e^(-0.5(x-mean)*(inv(variance))*(x-mean)'/sqrt(c))
                    where ' is transpose and * is matrix multiplication
            '''
            # TODO
            # - Comment/remove the exception
            # - Calculate the likelihood of sample x generated by this Gaussian
            # Note: use the described implementation of a Gaussian to ensure compatibility with the solutions
            # DONOT MODIFY CODE ABOVE THIS LINE
            x_minus_u = x - self.mean
            #p = np.exp(-0.5 * (x_minus_u).dot(self.inv).dot(np.transpose(x_minus_u))) / (np.sqrt(self.c))
            #p = np.exp(-0.5 * np.dot(x_minus_u, self.inv) * (x_minus_u)) / (np.sqrt(self.c))
            p = np.exp(-0.5 * np.sum((np.dot(x_minus_u, self.inv) * (x_minus_u)), axis=1)) / np.sqrt(self.c)
            # raise Exception('Impliment Guassian_pdf getLikelihood')
            # DONOT MODIFY CODE BELOW THIS LINE
            return p
