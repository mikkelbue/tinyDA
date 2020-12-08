import numpy as np
from .utils import  RecursiveSampleMoments

class GaussianRandomWalk:
    
    '''
    Standard MH random walk proposal.
    '''
    
    def __init__(self, C, scaling=1, adaptive=False, gamma=1.01, period=100):
        
        self.C = C
        self.d = self.C.shape[0]
        
        self._mean = np.zeros(self.d)
        self.scaling = scaling
        
        self.adaptive = adaptive
        
        if self.adaptive:
            self.k = 0
            self.gamma = gamma
            self.period = period
        
        self.t = 0
        
    def adapt(self, **kwargs):
        if self.adaptive:
            if self.t%self.period == 0:
                acceptance_rate = np.mean(kwargs['accepted'][-self.period:])
                self.scaling = np.exp(np.log(self.scaling) + self.gamma**-self.k*(acceptance_rate-0.24))
                self.k += 1
        else:
            pass
        
    def make_proposal(self, parameters):
        self.t += 1
        return parameters + self.scaling*np.random.multivariate_normal(self._mean, self.C)

    def get_acceptance_ratio(self, proposal_link, previous_link):
        return np.exp(proposal_link.posterior - previous_link.posterior)
        
class CrankNicolson(GaussianRandomWalk):
    
    ''' 
    This is the preconditioned Crank Nicolson proposal.
    '''
        
    def make_proposal(self, parameters):
        self.t += 1
        return np.sqrt(1 - self.scaling**2)*parameters + self.scaling*np.random.multivariate_normal(self._mean, self.C)

    def get_acceptance_ratio(self, proposal_link, previous_link):
        return np.exp(proposal_link.likelihood - previous_link.likelihood)


class AdaptiveMetropolis(GaussianRandomWalk):
    
    '''
    This is the Adaptive Metropolis proposal, according to Haario et al.
    '''
    
    def __init__(self, C0, t0=0, sd=None, epsilon=0):
        
        self.C = C0
        self.d = self.C.shape[0]
        
        self._mean = np.zeros(self.d)
        
        self.t0 = t0
        
        if sd is not None:
            self.sd = sd
        else:
            self.sd = min(1, 2.4**2/self.d)
        
        self.epsilon = epsilon
        
        self.t = 0
        
    def initialise_sampling_moments(self, parameters):
        self.AM_recursor = RecursiveSampleMoments(parameters,
                                                  np.zeros((self.d, self.d)),
                                                  sd=self.sd, 
                                                  epsilon=self.epsilon)
        
    def adapt(self, **kwargs):

        self.AM_recursor.update(kwargs['parameters'])
        
    def make_proposal(self, parameters):
        
        self.t += 1
        
        if self.t < self.t0:
            pass
        else:
            self.C = self.AM_recursor.get_sigma()
        
        return parameters + np.random.multivariate_normal(self._mean, self.C)



