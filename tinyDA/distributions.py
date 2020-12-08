# external imports
import numpy as np

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
