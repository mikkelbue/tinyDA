# external imports
import numpy as np
from scipy.linalg import sqrtm

# internal imports
from .utils import  RecursiveSampleMoments

class GaussianRandomWalk:
    
    '''
    Standard MH random walk proposal.
    '''
    
    def __init__(self, C, scaling=1, adaptive=False, gamma=1.01, period=100):
        
        # check if covariance operator is a square numpy array.
        if not isinstance(C, np.ndarray):
            raise TypeError('C must be a numpy array')
        elif C.ndim == 1:
            if not C.shape[0] == 1:
                raise ValueError('C must be an NxN array')
        elif not C.shape[0] == C.shape[1]:
            raise ValueError('C must be an NxN array')
        
        # set the covariance operator
        self.C = C
        
        # extract the dimensionality.
        self.d = self.C.shape[0]
        
        # set the distribution mean to zero.
        self._mean = np.zeros(self.d)
        
        # set the scaling.
        self.scaling = scaling
        
        # set adaptivity.
        self.adaptive = adaptive
        
        # if adaptive, set some adaptivity parameters
        if self.adaptive:
            
            # adaptivity counter for diminishing adaptivity.
            self.k = 0
            # adaptivity scaling.
            self.gamma = gamma
            # adaptivity period (delay between adapting)
            self.period = period
        
        # set a counter of how many times, the proposal has been called.
        self.t = 0
        
    def adapt(self, **kwargs):
        
        # if adaptive, run the adaptivity routines.
        if self.adaptive:
            
            # make sure the periodicity is repspected
            if self.t%self.period == 0:
                
                # compute the acceptance rate during the previous period.
                acceptance_rate = np.mean(kwargs['accepted'][-self.period:])
                # set the scaling so that the acceptance rate will converge to 0.24.
                self.scaling = np.exp(np.log(self.scaling) + self.gamma**-self.k*(acceptance_rate-0.24))
                # increase adaptivity counter for diminishing adaptivity.
                self.k += 1
        else:
            pass
        
    def make_proposal(self, link):
        # make a Gaussian RWMH proposal.
        self.t += 1
        return link.parameters + self.scaling*np.random.multivariate_normal(self._mean, self.C)

    def get_acceptance_ratio(self, proposal_link, previous_link):
        if np.isnan(proposal_link.posterior):
            return 0
        else:
            # get the acceptance probability.
            return np.exp(proposal_link.posterior - previous_link.posterior)
        
class CrankNicolson(GaussianRandomWalk):
    
    ''' 
    This is the preconditioned Crank Nicolson proposal, inheriting
    from the  GaussianRandomWalk.
    '''
        
    def make_proposal(self, link):
        # make a pCN proposal.
        self.t += 1
        return np.sqrt(1 - self.scaling**2)*link.parameters + self.scaling*np.random.multivariate_normal(self._mean, self.C)

    def get_acceptance_ratio(self, proposal_link, previous_link):
        if np.isnan(proposal_link.posterior):
            return 0
        else:
            # get the acceptance probability.
            return np.exp(proposal_link.likelihood - previous_link.likelihood)


class AdaptiveMetropolis(GaussianRandomWalk):
    
    '''
    This is the Adaptive Metropolis proposal, according to Haario et al.
    '''
    
    def __init__(self, C0, sd=None, epsilon=0, t0=0, period=100):
        
        # check if covariance operator is a square numpy array.
        if not isinstance(C0, np.ndarray):
            raise TypeError('C0 must be a numpy array')
        elif C0.ndim == 1:
            if not C0.shape[0] == 1:
                raise ValueError('C0 must be an NxN array')
        elif not C0.shape[0] == C0.shape[1]:
            raise ValueError('C0 must be an NxN array')
        
        # set the initial covariance operator.
        self.C = C0
        
        # extract the dimensionality.
        self.d = self.C.shape[0]
        
        # set a zero mean for the random draw.
        self._mean = np.zeros(self.d)
        
        # Set the scaling parameter for Diminishing Adaptation.
        if sd is not None:
            self.sd = sd
        else:
            self.sd = min(1, 2.4**2/self.d)
        
        # Set epsilon to avoid degeneracy.
        self.epsilon = epsilon
        
        # set the beginning of adaptation (rigidness of initial covariance).
        self.t0 = t0
        
        # Set the update period.
        self.period = period
        
        # set a counter of how many times, the proposal has been called.
        self.t = 0
        
    def initialise_sampling_moments(self, parameters):
        # initialise the sampling moments, which will compute the
        # adaptive covariance operator.
        self.AM_recursor = RecursiveSampleMoments(parameters,
                                                  np.zeros((self.d, self.d)),
                                                  sd=self.sd, 
                                                  epsilon=self.epsilon)
        
    def adapt(self, **kwargs):
        # AM is adaptive per definition. update the RecursiveSampleMoments
        # with the given parameters.
        self.AM_recursor.update(kwargs['parameters'])
        
        if self.t >= self.t0 and self.t%self.period == 0:
            self.C = self.AM_recursor.get_sigma()
        else:
            pass
        
    def make_proposal(self, link):
        self.t += 1
        # only use the adaptive proposal, if the initial time has passed.

        # make a proposal
        return link.parameters + np.random.multivariate_normal(self._mean, self.C)

class AdaptiveCrankNicolson(CrankNicolson):
    def __init__(self, C0, scaling=0.1, k=None, t0=0, period=100):
        
        # check if covariance operator is a square numpy array.
        if not isinstance(C0, np.ndarray):
            raise TypeError('C0 must be a numpy array')
        elif C0.ndim == 1:
            if not C0.shape[0] == 1:
                raise ValueError('C0 must be an NxN array')
        elif not C0.shape[0] == C0.shape[1]:
            raise ValueError('C0 must be an NxN array')
        
        # set the initial covariance operator.
        self.C = C0
        
        # extract the dimensionality.
        self.d = self.C.shape[0]
        
        # set a zero mean for the random draw.
        self._mean = np.zeros(self.d)
        
        # set the scaling.
        self.scaling = scaling
        
        # adaptive parameters
        self.B = self.C.copy()
        self.L = np.linalg.inv(self.C)
        self.operator = sqrtm(np.eye(self.d) - self.scaling**2*np.dot(self.B, self.L))
        
        # eigendecomposition of covariance matrix.
        self.alpha, self.e = np.linalg.eig(self.C)
        self.lamb = self.alpha.copy()
        
        # truncation.
        if k is not None:
            self.k = k
        else:
            self.k = self.d
        
        # set the beginning of adaptation (rigidness of initial covariance).
        self.t0 = t0
        
        # Set the update period.
        self.period = period
        
        # set a counter of how many times, the proposal has been called.
        self.t = 0
        
    def initialise_sampling_moments(self, parameters):
        self.x_n = np.dot(parameters, self.e)
        self.s_n = parameters**2
        self.lamb_n = (self.x_n - parameters)**2
        self.t += 1

    def adapt(self, **kwargs):
        # AM is adaptive per definition. update the RecursiveSampleMoments
        # with the given parameters.
        self.x_n = self.t/(self.t+1)*self.x_n + 1/(self.t+1)*np.dot(kwargs['parameters'], self.e)
        self.s_n = self.s_n + kwargs['parameters']**2
        self.lamb_n = 1/(self.t+1)*self.s_n - self.x_n**2
        #self.lamb_n =  self.t/(self.t+1)*self.lamb_n + 1/(self.t+1)*(self.x_n - kwargs['parameters'])**2
        
        if self.t >= self.t0 and self.t%self.period == 0:
            self.lamb[:self.k] = self.lamb_n[:self.k]
            self.lamb[self.lamb > self.alpha] = self.alpha[self.lamb > self.alpha]
            self.B = np.linalg.multi_dot((self.e, np.diag(self.lamb), self.e.T))
            self.operator = sqrtm(np.eye(self.d) - self.scaling**2*np.dot(self.B, self.L))
        else:
            pass        

    def make_proposal(self, link):
        self.t += 1
        # only use the adaptive proposal, if the initial time has passed.

        # make a proposal
        return np.dot(self.operator, link.parameters) + self.scaling*np.random.multivariate_normal(self._mean, self.B)
