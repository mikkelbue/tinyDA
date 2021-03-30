# external imports
import numpy as np

class CompositePrior:
    def __init__(self, distributions):
        self.distributions = distributions
        self.dim = len(distributions)
        
    def logpdf(self, x):
        return sum([self.distributions[i].logpdf(x[i]) for i in range(self.dim)])
        
    def rvs(self, n_samples=1):
        x = np.zeros((n_samples, self.dim))
        
        for i in range(self.dim):
            x[:,i] = self.distributions[i].rvs(size=n_samples)
        
        if n_samples == 1:
            return x.flatten()
        else:
            return x
        
    def ppf(self, x):
        for i in range(self.dim):
            x[:,i] = self.distributions[i].ppf(x[:,i])
        return x
            

class LogLike:
    '''
    LogLike is a minimal implementation of the (unnormalised) Gaussian
    likelihood function.
    '''
    
    def __init__(self, mean, covariance):
        
        # set the mean and covariance as attributes
        self.mean = mean
        self.cov = covariance
        
        # precompute the inverse of the covariance.
        self.cov_inverse = np.linalg.inv(self.cov)
        
    def logpdf(self, x):
        # compute the unnormalised likelihood.
        return -0.5*np.linalg.multi_dot(((x - self.mean).T, self.cov_inverse, (x - self.mean)))
        
class AdaptiveLogLike(LogLike):
    '''
    AdaptiveLogLike is a minimal implementation of the (unnormalised) Gaussian
    likelihood function, with offset, scaling, and rotation.
    '''
    def __init__(self, mean, covariance):
        super().__init__(mean, covariance)
        
        # set the initial bias.
        self.bias = np.zeros(self.mean.shape[0])
        
    def set_bias(self, bias, covariance_bias):
        # set the bias and the covariance of the bias.
        self.bias = bias
        self.cov_bias = covariance_bias
        
        # precompute the inverse.
        self.cov_inverse = np.linalg.inv(self.cov + self.cov_bias)
        
    def logpdf(self, x):
        # compute the unnormalised likelihood, with additional terms for offset, scaling, and rotation.
        return -0.5*np.linalg.multi_dot(((x + self.bias - self.mean).T, self.cov_inverse, (x + self.bias - self.mean)))
