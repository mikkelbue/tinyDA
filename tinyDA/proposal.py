# external imports
import warnings
import numpy as np
from scipy.linalg import sqrtm
import scipy.stats as stats

# internal imports
from .utils import RecursiveSampleMoments

class Proposal:
    '''
    Base proposal. Only used for inheritance.
    '''
    
    is_symmetric = False
    
    def __init__(self):
        pass
        
    def setup_proposal(self, **kwargs):
        pass
        
    def adapt(self, **kwargs):
        pass
        
    def make_proposal(self, link):
        pass
                
    def get_acceptance(self, proposal_link, previous_link):
        pass
        
    def get_q(self, x_link, y_link):
        pass

class IndependenceSampler(Proposal):
    
    '''
    Independence sampler using a proposal distribution q(x).
    
    Attributes
    ----------
    q : scipy.stats.rv_continuous
        A probability distribution to draw independent samples from. Usually a scipy disttribution,
        but the only requirement is that it has .rvs() and .logpdf() methods.
        
    Methods
    ----------
    setup_proposal(**kwargs)
        This proposal does not require any setup.
    adapt(**kwargs)
        This proposal is not adaptive.
    make_proposal(link)
        Generates an independent proposal from q.
    get_acceptance(proposal_link, previous_link)
        Computes the acceptance probability given a proposal link and the previous link.
    get_q(x_link, y_link)
        Computes the transition probability from x_link to y_link.
    '''
    
    is_symmetric = False
    
    def __init__(self, q):
        '''
        Parameters
        ----------
        q : scipy.stats.rv_continuous
            A probability distribution to draw independent samples from. Usually a scipy disttribution,
            but the only requirement is that it has .rvs() and .logpdf() methods.
        '''
        
        # set the proposal distribution.
        self.q = q
        
        # test that the proposal distribution has the required methods.
        try:
            self.q.logpdf(self.q.rvs(1))
        except AttributeError:
            raise
        
    def setup_proposal(self, **kwargs):
        pass
        
    def adapt(self, **kwargs):
        pass
        
    def make_proposal(self, link):
        
        # draw a random sample from the proposal distribution. 
        return self.q.rvs(1).flatten()
        
    def get_acceptance(self, proposal_link, previous_link):
        
        q_proposal = self.get_q(None, proposal_link)
        q_previous = self.get_q(None, previous_link)
        
        return np.exp(proposal_link.posterior - previous_link.posterior + q_previous - q_proposal)
        
    def get_q(self, x_link, y_link):
        
        # get the transition probability.:
        return self.q.logpdf(y_link.parameters)

class GaussianRandomWalk(Proposal):
    
    '''
    Standard Random Walk Metropolis Hastings proposal.
    
    Attributes
    ----------
    C : np.ndarray
        The covariance matrix of the proposal distribution.
    d : int
        The dimension of the target distribution.
    scaling : float
        The global scaling of the proposal.
    adaptive : bool
        Whether to adapt the global scaling of the proposal.
    gamma : float
        The adaptivity coefficient for global adaptive scaling.
    period : int
        How often to adapt the global scaling.
    k : int
        How many times the proposal has been adapted.
    t : int
        How many times the adapt method has been called.
        
    Methods
    ----------
    setup_proposal(**kwargs)
        This proposal does not require any setup.
    adapt(**kwargs)
        If adaptive=True, the proposal will adapt the global scaling.
    make_proposal(link)
        Generates a random walk proposal from the input link using the proposal covariance.
    get_acceptance(proposal_link, previous_link)
        Computes the acceptance probability given a proposal link and the previous link.
    '''
    
    is_symmetric = True
    
    def __init__(self, C, scaling=1, adaptive=False, gamma=1.01, period=100):
        '''
        Parameters
        ----------
        C : np.ndarray
            The covariance matrix of the proposal distribution.
        scaling : float, optional
            The global scaling of the proposal. Default is 1.
        adaptive : bool, optional
            Whether to adapt the global scaling of the proposal. Default is False.
        gamma : float, optional
            The adaptivity coefficient for the global adaptive scaling. Default is 1.01.
        period : int, optional
            How often to adapt the global scaling. Default is 100.
        '''
        
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
            
            # adaptivity scaling.
            self.gamma = gamma
            # adaptivity period (delay between adapting)
            self.period = period
            # initialise adaptivity counter for diminishing adaptivity.
            self.k = 0
        
        # initialise counter of how many times, adapt() has been called..
        self.t = 0
        
    def setup_proposal(self, **kwargs):
        pass
        
    def adapt(self, **kwargs):
        
        self.t += 1
        
        # if adaptive, run the adaptivity routines.
        if self.adaptive:
            
            # make sure the periodicity is respected
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
        return link.parameters + self.scaling*np.random.multivariate_normal(self._mean, self.C)

    def get_acceptance(self, proposal_link, previous_link):
        if np.isnan(proposal_link.posterior):
            return 0
        else:
            # get the acceptance probability.
            return np.exp(proposal_link.posterior - previous_link.posterior)
        
class CrankNicolson(GaussianRandomWalk):
    
    ''' 
    Preconditioned Crank Nicolson proposal. To use this proposal, the prior
    must be of the type scipy.stats.multivariate_normal.
    
    Attributes
    ----------
    C : np.ndarray
        The covariance matrix of the proposal distribution. This is set
        to the prior covariance, as required by pCN.
    d : int
        The dimension of the target distribution.
    scaling : float
        The global scaling ("beta") of the proposal.
    adaptive : bool
        Whether to adapt the global scaling of the proposal.
    gamma : float
        The adaptivity coefficient for the global adaptive scaling.
    period : int
        How often to adapt the global scaling.
    k : int
        How many times the proposal has been adapted.
    t : int
        How many times the adapt method has been called.
        
    Methods
    ----------
    setup_proposal(**kwargs)
        Sets the proposal covariance to the prior covariance, as required for pCN.
    adapt(**kwargs)
        If adaptive=True, the proposal will adapt the global scaling.
    make_proposal(link)
        Generates a pCN proposal from the input link using the proposal (prior) covariance.
    get_acceptance(proposal_link, previous_link)
        Computes the acceptance probability given a proposal link and the previous link.
    '''
    
    is_symmetric = False
    
    def __init__(self, scaling=0.1, adaptive=False, gamma=1.01, period=100):
        '''
        Parameters
        ----------
        scaling : float, optional
            The global scaling ("beta") of the proposal. Default is 0.1.
        adaptive : bool, optional
            Whether to adapt the global scaling of the proposal. Default is False.
        gamma : float, optional
            The adaptivity coefficient for the global adaptive scaling. Default is 1.01.
        period : int, optional
            How often to adapt the global scaling. Default is 100.
        '''
        
        # set the scaling.
        self.scaling = scaling
        
        # set adaptivity.
        self.adaptive = adaptive
        
        # if adaptive, set some adaptivity parameters
        if self.adaptive:
            
            # adaptivity scaling.
            self.gamma = gamma
            # adaptivity period (delay between adapting)
            self.period = period
            # initialise adaptivity counter for diminishing adaptivity.
            self.k = 0
        
        # initialise counter of how many times, adapt() has been called..
        self.t = 0
        
    def setup_proposal(self, **kwargs):
        
        # set the covariance operator
        self.C = kwargs['link_factory'].prior.cov
        
        # extract the dimensionality.
        self.d = self.C.shape[0]
        
        # set the distribution mean to zero.
        self._mean = np.zeros(self.d)
    
    def make_proposal(self, link):
        # make a pCN proposal.
        return np.sqrt(1 - self.scaling**2)*link.parameters + self.scaling*np.random.multivariate_normal(self._mean, self.C)

    def get_acceptance(self, proposal_link, previous_link):
        if np.isnan(proposal_link.posterior):
            return 0
        else:
            # get the acceptance probability.
            return np.exp(proposal_link.likelihood - previous_link.likelihood)
            
    def get_q(self, x_link, y_link):
        return stats.multivariate_normal.logpdf(y_link.parameters, mean=np.sqrt(1 - self.scaling**2)*x_link.parameters, cov=self.scaling**2*self.C)


class AdaptiveMetropolis(GaussianRandomWalk):
    
    '''
    Adaptive Metropolis proposal, according to Haario et al. (2001)
    
    Attributes
    ----------
    C : np.ndarray
        The covariance matrix of the proposal distribution.
    d : int
        The dimension of the target distribution.
    scaling : float
        The global scaling of the proposal.
    sd : float
        The AM scaling parameter.
    epsilon : float
        Parameter to prevent C from becoming singular.
    t0 : int
        When to start the adapting covariance matrix.
    adaptive : bool
        Whether to adapt the global scaling of the proposal.
    gamma : float
        The adaptivity coefficient for the global adaptive scaling.
    period : int
        How often to adapt the global scaling.
    k : int
        How many times the proposal has been globally adapted.
    t : int
        How many times the adapt method has been called.
        
    Methods
    ----------
    setup_proposal(**kwargs)
        A tinyDA.RecursiveSampleMoments is initialised to adapt the proposal covariance.
    adapt(**kwargs)
        The tinyDA.RecursiveSampleMoments is adapted using the latest sample.
        If adaptive=True, the proposal will also adapt the global scaling.
    make_proposal(link)
        Generates an Adaptive Metropolis proposal from the input link using the proposal covariance.
    get_acceptance(proposal_link, previous_link)
        Computes the acceptance probability given a proposal link and the previous link.
    '''
    
    def __init__(self, C0, sd=None, epsilon=1e-6, t0=0, period=100, adaptive=False, gamma=1.01):
        
        '''
        Parameters
        ----------
        C0 : np.ndarray
            The initial covariance matrix of the proposal distribution.
        sd : None or float, optional
            The AM scaling parameter. Default is None (compute from target dimensionality).
        epsilon : float, optional
            Parameter to prevent C from becoming singular. Must be small. Default is 1e-6.
        t0 : int, optional
            When to start adapting the covariance matrix. Default is 0 (start immediately).
        period : int, optional
            How often to adapt. Default is 100.
        adaptive : bool, optional
            Whether to adapt the global scaling of the proposal. Default is False.
        gamma : float, optional
            The adaptivity coefficient for the global adaptive scaling. Default is 1.01.
        '''
        
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
        
        # set the scaling factor.
        self.scaling = 1
        
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
        
        # set adaptivity.
        self.adaptive = adaptive
        
        # if adaptive, set some adaptivity parameters
        if self.adaptive:
            
            # adaptivity scaling.
            self.gamma = gamma
            # initialise adaptivity counter for diminishing adaptivity.
            self.k = 0
        
        # set a counter of how many times, adapt() has been called.
        self.t = 0
        

    def setup_proposal(self, **kwargs):
        # initialise the sampling moments, which will compute the
        # adaptive covariance operator.
        self.AM_recursor = RecursiveSampleMoments(kwargs['parameters'],
                                                  np.zeros((self.d, self.d)),
                                                  sd=self.sd, 
                                                  epsilon=self.epsilon)
        
    def adapt(self, **kwargs):
        
        super().adapt(**kwargs)
        
        # AM is adaptive per definition. update the RecursiveSampleMoments
        # with the given parameters.
        self.AM_recursor.update(kwargs['parameters'])
        
        if self.t >= self.t0 and self.t%self.period == 0:
            self.C = self.AM_recursor.get_sigma()
        else:
            pass

class AdaptiveCrankNicolson(CrankNicolson):
    
    '''
    Adaptive preconditioned Crank-Nicolson proposal, according to Hu and Yao (2016).
    
    Attributes
    ----------
    C : np.ndarray
        The initial covariance matrix of the proposal distribution.
    d : int
        The dimension of the target distribution.
    scaling : float
        The global scaling ("beta") of the proposal.
    J : int
        The truncation index, i.e. only dimensions below this index will be adapted.
    B : numpy.ndarray
        The adapted covariance matrix of the proposal distribution.
    L : numpy.ndarray
        Precision matrix of the prior, i.e. the inverse of the initial covariance matrix.
    operator : numpy.ndarray
        The adaptive pCN scaling operator. To avoid recomputing the scaling operator
        at each step, this array caches the result of sqrtm(eye(d) - beta**2*np.dot(B, L))
    t0 : int
        When to start adapting the covariance matrix.
    adaptive : bool
        Whether to adapt the global scaling of the proposal.
    gamma : float
        The adaptivity coefficient for the global adaptive scaling.
    period : int
        How often to adapt the global scaling.
    k : int
        How many times the proposal has been globally adapted.
    t : int
        How many times the adapt method has been called.
        
    Methods
    ----------
    setup_proposal(**kwargs)
        Setup the proposal to allow adaption according to Hu and Yao (2016).
    adapt(**kwargs)
        Use the latest sample to update the adaptive pCN operator.
        If adaptive=True, the proposal will also adapt the global scaling.
    make_proposal(link)
        Generates an adaptive pCN proposal from the input link using the proposal covariance.
    get_acceptance(proposal_link, previous_link)
        Computes the acceptance probability given a proposal link and the previous link.
    '''
    
    def __init__(self, scaling=0.1, J=None, t0=0, period=100, adaptive=False, gamma=1.01):
        '''
        Parameters
        ----------
        scaling : float, optional
            The global scaling ("beta") of the proposal. Default is 0.1.
        J : None or int, optional
            The truncation index, i.e. only dimensions below this index will be adapted.
            Default is None (adapt all dimensions).
        t0 : int, optional
            When to start adapting the covariance matrix. Default is 0 (start immediately).
        period : int, optional
            How often to adapt. Default is 100.
        adaptive : bool, optional
            Whether to adapt the global scaling of the proposal. Default is False.
        gamma : float, optional
            The adaptivity coefficient for the global adaptive scaling. Default is 1.01.
        '''
        
        # set the scaling.
        self.scaling = scaling
        
        # truncation.
        self.J = J
        
        # set the beginning of adaptation (rigidness of initial covariance).
        self.t0 = t0
        
        # Set the update period.
        self.period = period
        
        # set adaptivity.
        self.adaptive = adaptive
        
        # if adaptive, set some adaptivity parameters
        if self.adaptive:
            
            # adaptivity scaling.
            self.gamma = gamma
            # initialise adaptivity counter for diminishing adaptivity.
            self.k = 0
        
        # set a counter of how many times, adapt() has been called..
        self.t = 0
        
    def setup_proposal(self, **kwargs):
        
        # set the initial covariance operator.
        self.C = kwargs['link_factory'].prior.cov
        
        # extract the dimensionality.
        self.d = self.C.shape[0]
        
        # set a zero mean for the random draw.
        self._mean = np.zeros(self.d)
        
        # adaptive parameters
        self.B = self.C.copy()
        self.L = np.linalg.inv(self.C)
        self.operator = sqrtm(np.eye(self.d) - self.scaling**2*np.dot(self.B, self.L))
        
        # eigendecomposition of covariance matrix.
        self.alpha, self.e = np.linalg.eig(self.C)
        self.lamb = self.alpha.copy()
        
        # truncation.
        if self.J is None:
            self.J = self.d
        
        u_j = np.inner(kwargs['parameters'], self.e.T)
        self.x_n = u_j
        self.lamb_n = (self.x_n - u_j)**2

    def adapt(self, **kwargs):
        
        super().adapt(**kwargs)
        
        # ApCN is adaptive per definition. update the moments.
        u_j = np.inner(kwargs['parameters'], self.e.T)
        self.x_n = self.t/(self.t+1)*self.x_n + 1/(self.t+1)*u_j
        self.lamb_n = self.t/(self.t+1)*self.lamb_n + 1/(self.t+1)*(self.x_n - u_j)**2
        
        # compute the operator, if the initial adaptation is complete and
        # the period matches.
        if self.t >= self.t0 and self.t%self.period == 0:
            self.lamb[:self.J] = self.lamb_n[:self.J]
            self.lamb[self.lamb > self.alpha] = self.alpha[self.lamb > self.alpha]
            self.B = np.linalg.multi_dot((self.e, np.diag(self.lamb), self.e.T))
            self.operator = sqrtm(np.eye(self.d) - self.scaling**2*np.dot(self.B, self.L))
        else:
            pass        

    def make_proposal(self, link):
        # only use the adaptive proposal, if the initial time has passed.

        # make a proposal
        return np.dot(self.operator, link.parameters) + self.scaling*np.random.multivariate_normal(self._mean, self.B)
        
    def get_q(self, x_link, y_link):
        return stats.multivariate_normal.logpdf(y_link.parameters, mean=np.dot(self.operator, x_link.parameters), cov=self.scaling**2*self.B)

class SingleDreamZ(GaussianRandomWalk):
    
    '''
    Dream(Z) proposal, similar to the DREAM(ZS) algorithm (see e.g. Vrugt 2016).
    
    Attributes
    ----------
    M0 : int
        Size of the initial archive.
    d : int
        The dimension of the target distribution.
    scaling : float
        The global scaling of the proposal.
    delta : int
        Number of sample pairs from the archive to use to compute the jumping direction.
    b : float
        Upper and lower bound for the uniform pertubation distribution, i.e. e ~ U(-b,b).
    b_star : float
        Scale for the Gaussian pertubation distribution, i.e. epsilon ~ N(0, b_star).
    Z_method : str
        How to draw the initial archive. 
        Can be 'random' for simple random sampling or 'lhs' for latin hypercube sampling.
    nCR : int
        Size of the crossover probability distribution.
    adaptive : bool
        Whether to adapt the global scaling and crossover distribution of the proposal.
    gamma : float
        The adaptivity coefficient for global adaptive scaling.
    period : int
        How often to adapt the global scaling and crossover probabilities.
    k : int
        How many times the proposal has been adapted.
    t : int
        How many times the adapt method has been called.
        
    Methods
    ----------
    setup_proposal(**kwargs)
        The proposal creates the initial archive from samples from the prior, 
        using the specified Z-method. If set to "lhs" and pyDOE is not available, 
        it will fall back to simple random sampling.
    adapt(**kwargs)
        If adaptive=True, the proposal will adapt the global scaling and the
        crossover distribution.
    make_proposal(link)
        Generates a SingleDreamZ proposal.
    get_acceptance(proposal_link, previous_link)
        Computes the acceptance probability given a proposal link and the previous link.
    '''
    
    def __init__(self, M0, delta=1, b=5e-2, b_star=1e-6, Z_method='random', nCR=3, adaptive=False, gamma=1.01, period=100):
        
        '''
        Parameters
        ----------
        M0 : np.ndarray
            Size of the initial archive.
        delta : int, optional
            Number of sample pairs from the archive to use to compute the jumping direction. 
            Default is 1.
        b : float, optional
            Upper and lower bound for the uniform pertubation distribution, i.e. e ~ U(-b,b).
            Default is 5e-2.
        b_star : float, optional
            Scale for the Gaussian pertubation distribution, i.e. epsilon ~ N(0, b_star).
            Must be small. Default is 1e-6.
        Z_method : str, optional
            How to draw the initial archive. 
            Can be 'random' for simple random sampling or 'lhs' for latin hypercube sampling.
            Default is 'random'.
        nCR : int, optional
            Size of the crossover probability distribution. Default is 3.
        adaptive : bool, optional
            Whether to adapt the global scaling of the proposal and crossover distribution. 
            Default is False.
        gamma : float, optional
            The adaptivity coefficient for global adaptive scaling. Default is 1.01.
        period : int, optional
            How often to adapt the global scaling and crossover distribution. Default is 100.
        '''
        
        warnings.warn(' SingleDreamZ is an EXPERIMENTAL proposal, similar to the DREAM(ZS) algorithm (see e.g. Vrugt 2016), but using only a single chain.\n')
        
        # set initial archive size.
        self.M = M0
        
        # set global scaling
        self.scaling = 1
        
        # set DREAM parameters
        self.delta = delta
        self.b = b
        self.b_star = b_star
        
        # Set the method to create the initial archive.
        self.Z_method = Z_method
        
        # set adaptivity.
        self.adaptive = adaptive
        
        # set crossover distribution.
        self.nCR = nCR
        self.mCR = None
        self.pCR = np.array(self.nCR * [1/self.nCR])
        
        # if adaptive, set some adaptivity parameters
        if self.adaptive:
            
            # adaptivity scaling.
            self.gamma = gamma
            # adaptivity period (delay between adapting)
            self.period = period
            # initialise adaptivity counter for diminishing adaptivity.
            self.k = 0
            
            # DREAM-specifivc adaptivity
            self.LCR = np.zeros(self.nCR)
            self.DeltaCR = np.ones(self.nCR)
        
        self.t = 0
        
    def setup_proposal(self, **kwargs):
        
        prior = kwargs['link_factory'].prior
        
        # get the dimension and the initial scaling.
        self.d = prior.dim
        
        # draw initial archive with latin hypercube sampling.
        if self.Z_method == 'lhs':
            
            try:
                # try to import pyDOE and transform the samples to the prior distribution.
                from pyDOE import lhs
                self.Z = lhs(self.d, samples=self.M)
                self.Z = prior.ppf(self.Z)
                return
                
            except ModuleNotFoundError:
                # if pyDOE is not installed, fall back on simple random sampling.
                warnings.warn(' pyDOE module not found. Falling back on default Z-sampling method: \'random\'.\n')
                pass
                
            except AttributeError:
                # if the prior is a multivariate_normal, it will not have a .ppf-method.
                if isinstance(prior, stats._multivariate.multivariate_normal_frozen):
                    # instead draw samples from indpendent normals, according to the prior means and variances.
                    for i in range(self.d):
                        self.Z[:, i] = stats.norm(loc=prior.mean[i], scale=prior.cov[i,i]).ppf(self.Z[:, i])
                    return
                
                else:
                    # if the prior does not have a .ppf-method, and it's not a multivariate gaussian, fall back on simple random sampling.
                    warnings.warn(' Prior does not have .ppf method. Falling back on default Z-sampling method: \'random\'.\n')
                    pass
        
        # do simple random sampling from the prior.
        self.Z = prior.rvs(self.M)
        
    def adapt(self, **kwargs):
        
        super().adapt(**kwargs)
        
        # extend the archive.
        self.Z = np.vstack((self.Z, kwargs['parameters']))
        self.M = self.Z.shape[0]
        
        # adaptivity
        if self.adaptive and self.t%self.period == 0:
            
            # compute new multinomial distribution according to the normalised jumping distance.
            self.DeltaCR[self.mCR] = self.DeltaCR[self.mCR] + (kwargs['jumping_distance']**2/np.std(self.Z, axis=0)**2).sum()
            self.LCR[self.mCR] = self.LCR[self.mCR] + 1
            
            if np.all(self.LCR > 0):
                DeltaCR_mean = self.DeltaCR/self.LCR
                self.pCR = DeltaCR_mean/DeltaCR_mean.sum()
        
    def make_proposal(self, link):
        
        # initialise the jump vectors.
        Z_r1 = np.zeros(self.d)
        Z_r2 = np.zeros(self.d)
        
        # get jump vector components.
        for i in range(self.delta):
            r1, r2 = np.random.choice(self.M, 2, replace=False)
            Z_r1 += self.Z[r1,:]
            Z_r2 += self.Z[r2,:]
        
        # randomly choose crossover probability.
        self.mCR = np.random.choice(self.nCR, p=self.pCR)
        CR = (self.mCR+1) / self.nCR
        
        # set up the subspace indicator, deciding which dimensions to pertubate.
        subspace_indicator = np.zeros(self.d)
        subspace_draw = np.random.uniform(size=self.d)
        subspace_indicator[subspace_draw < CR] = 1

        # if no dimensions were chosen, pick one a random.
        if subspace_indicator.sum() == 0:
            subspace_indicator[np.random.choice(self.d)] = 1
        
        # compute the optimal scaling.
        gamma_DREAM = self.scaling*2.38/np.sqrt(2*self.delta*subspace_indicator.sum())
        
        # get the random scalings and gaussian pertubation.
        e = np.random.uniform(-self.b, self.b, size=self.d)
        epsilon = np.random.normal(0, self.b_star, size=self.d)
        
        return link.parameters + subspace_indicator*((np.ones(self.d) + e)*gamma_DREAM*(Z_r1 - Z_r2) + epsilon)
