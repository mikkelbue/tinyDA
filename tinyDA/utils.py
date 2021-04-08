import multiprocessing
import multiprocessing.pool

import numpy as np

class RecursiveSampleMoments:
    """
    Iteratively constructs a sample mean and covariance, given input samples.
    Used to capture an estimate of the mean and covariance of the bias of an MLDA
    coarse model, and for the Adaptive Metropolis proposal.
    """

    def __init__(self, mu0, sigma0, t=1, sd=1, epsilon=0):
        
        self.mu = mu0
        self.d = self.mu.shape[0]
        
        self.sigma = sigma0
        
        self.t = t
        
        self.sd = sd
        self.epsilon = epsilon

    def __call__(self):
        return self.mu, self.sigma

    def get_mu(self):
        # Returns the current mu value
        return self.mu

    def get_sigma(self):
        #Returns the current covariance value
        return self.sigma

    def update(self, x):
        # Updates the mean and covariance given a new sample x
        mu_previous = self.mu.copy()

        self.mu = (1 / (self.t + 1)) * (self.t * mu_previous + x)

        self.sigma = (self.t-1)/self.t * self.sigma + self.sd/self.t * (
            self.t * np.outer(mu_previous, mu_previous)
            - (self.t + 1) * np.outer(self.mu, self.mu)
            + np.outer(x, x) + self.epsilon*np.eye(self.d)
        )

        self.t += 1

class ZeroMeanRecursiveSampleMoments:
    """
    Iteratively constructs a sample covariance, with zero mean given input samples.
    It is a specialised version of RecursiveSampleMoments, used only in the state
    dependent error model.
    """

    def __init__(self, sigma0, t=1):
        
        self.sigma = sigma0
        self.d = self.sigma.shape[0]
        
        self.t = t

    def __call__(self):
        return self.sigma

    def get_sigma(self):
        # Returns the current covariance value
        return self.sigma

    def update(self, x):
        # Updates the covariance given a new sample x

        self.sigma = (self.t-1)/self.t * self.sigma + 1/self.t * np.outer(x, x)

        self.t += 1

class NoDaemonProcess(multiprocessing.Process):
    @property
    def daemon(self):
        return False

    @daemon.setter
    def daemon(self, value):
        pass

class NoDaemonContext(type(multiprocessing.get_context())):
    Process = NoDaemonProcess

# We sub-class multiprocessing.pool.Pool instead of multiprocessing.Pool
# because the latter is only a wrapper function, not a proper class.
class NestablePool(multiprocessing.pool.Pool):
    def __init__(self, *args, **kwargs):
        kwargs['context'] = NoDaemonContext()
        super(NestablePool, self).__init__(*args, **kwargs)
