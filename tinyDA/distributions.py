import numpy as np
from scipy.stats import multivariate_normal

class LogMVNormal:
    def __init__(self, mean, covariance):
        self.distribution = multivariate_normal(mean=mean, cov=covariance)
        
    def pdf(self, parameters):
        return np.log(self.distribution.pdf(parameters))
    
    def rvs(self, size=1):
        return self.distribution.rvs(size=size)
        
class LogLike:
    def __init__(self, data, covariance):
        self.data = data
        self.cov = covariance
        self.cov_inverse = np.linalg.inv(self.cov)
        
    def pdf(self, model_output):
        return -0.5*np.linalg.multi_dot(((model_output - self.data).T, self.cov_inverse, (model_output - self.data)))
        
class AdaptiveLogLike(LogLike):
    def __init__(self, data, covariance):
        super().__init__(data, covariance)
        self.bias = np.zeros(self.data.shape[0])
        
    def set_bias(self, bias, covariance_bias):
        self.bias = bias
        self.cov_bias = covariance_bias
        self.cov_inverse = np.linalg.inv(self.cov + self.cov_bias)
        
    def pdf(self, model_output):
        return -0.5*np.linalg.multi_dot(((model_output + self.bias - self.data).T, self.cov_inverse, (model_output + self.bias - self.data)))
