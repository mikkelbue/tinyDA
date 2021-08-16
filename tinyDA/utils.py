import numpy as np
from scipy.optimize import minimize

class RecursiveSampleMoments:
    '''
    Iteratively constructs a sample mean and covariance, given input samples.
    Used to capture an estimate of the mean and covariance of the bias of an MLDA
    coarse model, and for the Adaptive Metropolis (AM) proposal.
    
    Attributes
    ----------
    mu : numpy.ndarray
        The mean array.
    sigma : numpy.ndarray
        The covariance matrix.
    d : int
        The dimensionality.
    t : int
        The sample size, i.e. iteration counter.
    sd : float
        The AM scaling parameter.
    epsilon : float
        Parameter to prevent C from becoming singular (used for AM).
        
    Methods
    ----------
    get_mu()
        Returns the current mean.
    get_sigma()
        Returns the current covariance matrix.
    update(x)
        Update the sample moments with an input array x.
    '''

    def __init__(self, mu0, sigma0, t=1, sd=1, epsilon=0):
        '''
        Parameters
        ----------
        mu0 : numpy.ndarray
            The initial mean array.
        sigma0 : numpy.ndarray
            The initial covariance matrix.
        t : int, optional
            The initial sample size, i.e. iteration counter. Default is 1.
        sd : float, optional
            The AM scaling parameter. Default is 1
        epsilon : float, optional
            Parameter to prevent C from becoming singular (used for AM).
            Default is 0.
        
        '''
        
        # set the initial mean and dimensionality
        self.mu = mu0
        self.d = self.mu.shape[0]
        
        # set the initial covariance matrix.
        self.sigma = sigma0
        
        # set the counter
        self.t = t
        
        # set AM-specific parameters.
        self.sd = sd
        self.epsilon = epsilon

    def __call__(self):
        '''
        Returns
        ----------
        tuple
            Returns tuple of the current (mean, covariance), each a numpy.ndarray.
        '''
        
        return self.mu, self.sigma

    def get_mu(self):
        '''
        Returns
        ----------
        numpy.ndarray
            Returns the current mean.
        '''
        
        # Returns the current mu value
        return self.mu

    def get_sigma(self):
        '''
        Returns
        ----------
        numpy.ndarray
            Returns the current covariance.
        '''
        
        #Returns the current covariance value
        return self.sigma

    def update(self, x):
        '''
        Parameters
        ----------
        x : numpy.ndarray
            Updates the sample moments using an input array x.
        '''
        
        # Updates the mean and covariance given a new sample x
        mu_previous = self.mu.copy()

        self.mu = (1 / (self.t + 1)) * (self.t * mu_previous + x)

        self.sigma = (self.t-1)/self.t * self.sigma + self.sd/self.t * (
            self.t * np.outer(mu_previous, mu_previous)
            - (self.t + 1) * np.outer(self.mu, self.mu)
            + np.outer(x, x) + self.epsilon*np.eye(self.d)
        )

        self.t += 1

class ZeroMeanRecursiveSampleMoments(RecursiveSampleMoments):
    '''
    Iteratively constructs a sample covariance, with zero mean given input samples.
    It is a specialised version of RecursiveSampleMoments, used only in the state
    dependent error model.
    
    Attributes
    ----------
    sigma : numpy.ndarray
        The covariance matrix.
    d : int
        The dimensionality.
    t : int
        The sample size, i.e. iteration counter.
        
    Methods
    ----------
    get_sigma()
        Returns the current covariance matrix.
    update(x)
        Update the sample moments with an input array x.
    '''

    def __init__(self, sigma0, t=1):
        '''
        Parameters
        ----------
        sigma0 : numpy.ndarray
            The initial covariance matrix.
        t : int, optional
            The initial sample size, i.e. iteration counter. Default is 1.
        '''
        
        self.sigma = sigma0
        self.d = self.sigma.shape[0]
        
        self.t = t

    def __call__(self):
        '''
        Returns
        ----------
        numpy.ndarray
            Returns the current covariance.
        '''
        
        return self.sigma
        
    def get_mu(self):
        pass

    def get_sigma(self):
        '''
        Returns
        ----------
        numpy.ndarray
            Returns the current covariance.
        '''

        # Returns the current covariance value
        return self.sigma

    def update(self, x):
        '''
        Parameters
        ----------
        x : numpy.ndarray
            Updates the covariance using an input array x.
        '''

        # Updates the covariance given a new sample x

        self.sigma = (self.t-1)/self.t * self.sigma + 1/self.t * np.outer(x, x)

        self.t += 1
        
class RandomFourierGaussianDensity:
    def __init__(self, sigma, n_mc, dim, lmbda=1e-3):
        
        # internalise parameters
        self.sigma = sigma
        self.n_mc = n_mc
        self.dim = dim
        self.lmbda = lmbda
        
        # get Monte Carlo integration points.
        self.omega = np.random.normal(scale=self.sigma**-2, size=(self.dim, self.n_mc))
        self.u = np.random.uniform(0, 2*np.pi, size=self.n_mc)
        
        # set the initial parameters
        self.theta = np.zeros(n_mc)
        
        # initialise unnormlised objective matrices.
        self.b_bar = np.zeros(self.n_mc)
        self.C_bar = np.zeros((self.n_mc, self.n_mc))
        
        # initialise counter
        self.t = 0
        
    def phi(self, x):
        return np.sqrt(2/self.n_mc)*np.cos(np.dot(x, self.omega) + self.u)

    def dphi(self, x, d):
        return -np.sqrt(2/self.n_mc)*np.sin(np.dot(x, self.omega) + self.u) * self.omega[d,:]

    def ddphi(self, x, d):
        return -np.sqrt(2/self.n_mc)*np.cos(np.dot(x, self.omega) + self.u) * self.omega[d,:]**2
    
    def update(self, x):
        
        self.t += 1
        
        for d in range(self.dim):
            self.b_bar += self.ddphi(x, d)
            
            temp = self.dphi(x, d)
            self.C_bar += np.outer(temp, temp) 
        
    def fit(self):
        b = -1/self.t*self.b_bar
        C = 1/self.t*self.C_bar
        self.theta = np.linalg.solve(C + self.lmbda*np.eye(self.n_mc), b)
            
    def get_gradient(self, x):
        gradient = np.zeros((self.dim, self.n_mc))
        for d in range(self.dim):
            gradient[d,:] = self.dphi(x, d)
        return np.dot(gradient, self.theta)


def get_MAP(link_factory, initial_parameters=None, **kwargs):
    '''
    Returns the Maximum a Posteriori estimate of a link factory.
    
    Parameters
    ----------
    link_factory : tinyDA.LinkFactory
        The link factory to use for computing the MAP point.
    initial_parameters : numpy.ndarray, optional
        The starting point for the optimisation. Default is None
        (random draw from the prior).
    **kwargs : optional
        Keyword arguments passed to scipy.optimize.minimize.
    
    Returns
    ----------
    numpy.ndarray
        Maximum a Posteriori estimate.
    '''
    
    if initial_parameters is None:
        initial_parameters = link_factory.prior.rvs()
    
    negative_log_posterior = lambda parameters: -link_factory.create_link(parameters).posterior
    MAP = minimize(negative_log_posterior, initial_parameters, **kwargs)
    return MAP['x']
    

def get_ML(link_factory, initial_parameters=None, **kwargs):
    '''
    Returns the Maximum Likelihood estimate of a link factory.
    
    Parameters
    ----------
    link_factory : tinyDA.LinkFactory
        The link factory to use for computing the ML point.
    initial_parameters : numpy.ndarray, optional
        The starting point for the optimisation. Default is None
        (random draw from the prior).
    **kwargs : optional
        Keyword arguments passed to scipy.optimize.minimize.
    
    Returns
    ----------
    numpy.ndarray
        Maximum Likelihood estimate.
    '''
    
    if initial_parameters is None:
        initial_parameters = link_factory.prior.rvs()
    
    negative_log_likelihood = lambda parameters: -link_factory.create_link(parameters).likelihood
    ML = minimize(negative_log_likelihood, initial_parameters, **kwargs)
    return ML['x']
