# external imports
import warnings
try:
    import ray.util.multiprocessing as mp
except ModuleNotFoundError:
    import multiprocessing as mp
import numpy as np
from scipy.linalg import sqrtm
import scipy.stats as stats
from scipy.special import logsumexp

try:
    import TransportMaps as tm
    from .transportmap import DataDist
except ModuleNotFoundError:
    pass

# internal imports
from .link import DummyLink
from .utils import RecursiveSampleMoments
from .transportmap import DataDist, get_gaussian_transport_map

class IndependenceSampler:
    
    '''
    Independence sampler using a proposal distribution q(x).
    '''
    
    is_symmetric = False
    
    def __init__(self, q):
        
        self.q = q
        
    def setup_proposal(self, **kwargs):
        pass
        
    def adapt(self, **kwargs):
        pass
        
    def make_proposal(self, link):
        return self.q.rvs(1).flatten()
        
    def get_acceptance(self, proposal_link, previous_link):
        
        if hasattr(self.q, 'logpdf'):
            q_proposal = self.q.logpdf(proposal_link.parameters)
            q_previous = self.q.logpdf(proposal_link.parameters)
        elif hasattr(self.q, 'log_pdf'):
            q_proposal = self.q.log_pdf(np.expand_dims(proposal_link.parameters, axis=0))
            q_previous = self.q.log_pdf(np.expand_dims(proposal_link.parameters, axis=0))
        else:
            raise AttributeError('Proposal distribution has neither .logpdf() or log_pdf() method')
        
        return np.exp(proposal_link.posterior - previous_link.posterior + q_previous - q_proposal)
        
    def get_q(self, x_link, y_link):
        if hasattr(self.q, 'logpdf'):
            return self.q.logpdf(y_link.parameters)
        elif hasattr(self.q, 'log_pdf'):
            return self.q.log_pdf(np.expand_dims(y_link.parameters, axis=0))
        else:
            raise AttributeError('Proposal distribution has neither .logpdf() or log_pdf() method')

class GaussianRandomWalk:
    
    '''
    Standard MH random walk proposal.
    '''
    
    is_symmetric = True
    
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
    This is the preconditioned Crank Nicolson proposal, inheriting
    from the  GaussianRandomWalk.
    '''
    
    is_symmetric = False
    
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
    This is the Adaptive Metropolis proposal, according to Haario et al.
    '''
    
    def __init__(self, C0, sd=None, epsilon=1e-6, t0=0, period=100, adaptive=False, gamma=1.01):
        
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
        self.t += 1
        
    def adapt(self, **kwargs):
        # AM is adaptive per definition. update the RecursiveSampleMoments
        # with the given parameters.
        self.AM_recursor.update(kwargs['parameters'])
        
        super().adapt(**kwargs)
        
        if self.t >= self.t0 and self.t%self.period == 0:
            self.C = self.AM_recursor.get_sigma()
        else:
            pass

class AdaptiveCrankNicolson(CrankNicolson):
    
    '''
    Adaptive preconditioned Crank-Nicolson proposal, see Hu and Yao (2016).
    '''
    
    def __init__(self, C0, scaling=0.1, J=None, t0=0, period=100, adaptive=False, gamma=1.01):
        
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
        if J is not None:
            self.J = J
        else:
            self.J = self.d
        
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
        u_j = np.inner(kwargs['parameters'], self.e.T)
        self.x_n = u_j
        self.lamb_n = (self.x_n - u_j)**2
        self.t += 1

    def adapt(self, **kwargs):
        # ApCN is adaptive per definition. update the moments.
        u_j = np.inner(kwargs['parameters'], self.e.T)
        self.x_n = self.t/(self.t+1)*self.x_n + 1/(self.t+1)*u_j
        self.lamb_n =  self.t/(self.t+1)*self.lamb_n + 1/(self.t+1)*(self.x_n - u_j)**2
        
        super().adapt(**kwargs)
        
        # commpute the operator, if the initial adaptation is complete and
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
    '''
    
    def __init__(self, M0, delta=1, b=5e-2, b_star=1e-6, Z_method='random', nCR=3, adaptive=False, gamma=1.01, period=100):
        
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
                    
            except ModuleNotFoundError:
                # if pyDOE is not installed, fall back on simple random sampling.
                warnings.warn(' pyDOE module not found. Falling back on default Z-sampling method: \'random\'.\n')
                pass
        
        # do simple random sampling from the prior.
        self.Z = prior.rvs(self.M)
        
    def adapt(self, **kwargs):
        
        # extend the archive.
        self.Z = np.vstack((self.Z, kwargs['parameters']))
        self.M = self.Z.shape[0]
        
        super().adapt(**kwargs)
        
        # adaptivity
        if self.adaptive:
            
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
                
class GaussianTransportMap:
    
    '''
    Transport Map enhanced Gaussian Random Walk, or independence sampler
    (set flag independence_sampler=True). See Parno and Marzouk (2017).
    '''
    
    def __init__(self, kernel, order=1, t0=1000, period=100, independence_sampler=False, T0=None, reg_alpha=1, discard_fraction=0.9, k_history=None):
        
        try:
            print('Loaded TransportMaps v{}'.format(tm.__version__))
        except NameError as e:
            raise ModuleNotFoundError("No module named 'TransportMaps'. GaussianTransportMap proposal is not available.") from e
            
        if isinstance(kernel, CrankNicolson) or isinstance(kernel, AdaptiveMetropolis) or isinstance(kernel, SingleDreamZ):
            if not independence_sampler:
                raise NotImplementedError('GaussianTransportMap must have independence_sampler=True for kernel: {}'.format(type(kernel)))
            if T0 is not None:
                raise NotImplementedError('GaussianTransportMap must have T0=None for kernel: {}'.format(type(kernel)))
            if isinstance(kernel, AdaptiveMetropolis) or isinstance(kernel, AdaptiveCrankNicolson):
                if t0 < kernel.t0:
                    raise ValueError('Adaptive kernels should start adapting before independence sampler kicks in: t0 >> kernel.t0.')
        
        warnings.warn(' GaussianTransportMap is an EXPERIMENTAL proposal. Use with caution.\n')
            
        # set the kernel.
        self.kernel = kernel
        
        # set the map order
        self.order = order
        
        if self.order == 1:
            pass
        elif self.order == 2:
            warnings.warn(' Inversion of second order transport maps can be slow, particularly for high dimensional maps.\n')
        else:
            raise ValueError('Transport maps can be of first or second order only')
        
        # set the beginning of adaptation (rigidness of initial covariance).
        self.t0 = t0
        
        # Set the update period.
        self.period = period
        
        # set independence sampler flag.
        self.independence_sampler = independence_sampler
        
        # set the initial transport map
        self.T0 = T0
        
        # set the regularisation parameter
        self.reg_alpha = reg_alpha
        
        # set the discard fraction
        self.discard_fraction = discard_fraction
        
        # set the size of the history sample used to generate the map.
        self.k_history = k_history
        if self.k_history is None:
            warnings.warn(' Using entire sampling history to generate transport map. This can be memory intensive.\n')
        
        # initialise counter of how many times, adapt() has been called..
        self.t = 0
        
    def setup_proposal(self, **kwargs):
        
        # setup the kernel
        self.kernel.setup_proposal(**kwargs)
        
        # get the dimensionality
        self.d = self.kernel.d
        
        # set the reference distribution
        self.rho = tm.Distributions.GaussianDistribution(np.zeros(self.d), np.eye(self.d))
        
        # set up an initial (dummy) transport map
        if self.T0 is not None:
            self.T = self.T0
        else:
            self.T = tm.Maps.IdentityTransportMap(self.d)
        
        # initialise the history
        self.history = kwargs['parameters']
        
        # set the initial guess for the next KL minimisation.
        self.map_coeffs = None
    
    def adapt(self, **kwargs):
        
        self.t += 1
        
        # add the latest parameters to the archive
        self.history = np.vstack((self.history, kwargs['parameters']))
        
        if not self.independence_sampler or self.t < self.t0:
            self.kernel.adapt(**kwargs)
        
        # truncate the archive to make sure burnin is discarded.
        if self.t == self.t0:
            self.history = self.history[int(self.discard_fraction*self.history.shape[0]):,:]
        
        # do the transport map adaptation.
        if self.t >= self.t0 and self.t%self.period == 0:
            
            # set up a distribution for the arhive.
            if self.k_history is None:
                data = self.history
            else:
                idx = np.random.randint(0, self.history.shape[0], self.k_history)
                data = self.history[idx, :]
            
            # get the transport map and coefficients.
            self.T, coeffs = get_gaussian_transport_map(data, self.order, self.reg_alpha, self.map_coeffs)
            
            # expand coefficient dimension.
            self.map_coeffs = np.expand_dims(coeffs, axis=1)
        
    def make_proposal(self, link):
        
        # if independece sampler flag is set, and map has been built,
        # dram an independent sample
        if self.independence_sampler and self.t >= self.t0:
            
            # draw proposal from rho.
            r_proposal = self.rho.rvs(1).flatten()
            
            return self.T.inverse(np.expand_dims(r_proposal, axis=0)).flatten()
        
        # otherwise
        else:
            
            # push the parameters to the reference distribution.
            r_previous = self.T(np.expand_dims(link.parameters, axis=0)).flatten()
        
            # create a reference proposal.
            r_proposal = self.kernel.make_proposal(DummyLink(r_previous))
        
        # push the reference proposal back through the map and return it.
        return self.T.inverse(np.expand_dims(r_proposal, axis=0)).flatten()

        
    def get_acceptance(self, proposal_link, previous_link):
        
        if np.isnan(proposal_link.posterior):
            return 0
        
        else:
            # if using the independence sampler, get the reference parameters and density first.
            if self.independence_sampler and self.t >= self.t0:
                
                # get the reference parameters.
                r_proposal = self.T(np.expand_dims(proposal_link.parameters, axis=0))
                r_previous = self.T(np.expand_dims(previous_link.parameters, axis=0))
                
                # get the densities.
                q_y = self.rho.log_pdf(r_proposal)
                q_x = self.rho.log_pdf(r_previous)
                
                # NB! it is cheaper to take the log_det_grad of the parameters,
                # than the inverse of the reference parameters, but the sign is flipped.
                # see See Parno and Marzouk (2017) for details.
                return np.exp(proposal_link.posterior - previous_link.posterior + q_x - q_y \
                    - self.T.log_det_grad_x(np.expand_dims(proposal_link.parameters, axis=0)) \
                    + self.T.log_det_grad_x(np.expand_dims(previous_link.parameters, axis=0)))
            
            # otherwise, return the transport map acceptance rate (for a symmetric proposal!).
            # for the CrankNicolson kernel, the second factor is always one, since T is identity 
            # when t<t0, and when t>t0, the independence sampler is used.
            else:
                return self.kernel.get_acceptance(proposal_link, previous_link) \
                    * np.exp(- self.T.log_det_grad_x(np.expand_dims(proposal_link.parameters, axis=0)) \
                             + self.T.log_det_grad_x(np.expand_dims(previous_link.parameters, axis=0)))

class MultipleTry:
    
    '''
    Multiple-Try proposal, which will take any other TinyDA proposal
    as a kernel. The parameter k sets the number of tries.
    '''
    
    is_symmetric = True
    
    def __init__(self, kernel, k):
        
        # set the kernel
        self.kernel = kernel
        
        # set the number of tries per proposal.
        self.k = k
        
        if self.kernel.adaptive:
            warnings.warn(' Using global adaptive scaling with MultipleTry proposal can be unstable.\n')
            
    def open_pool(self):
        self.pool = mp.Pool(self.k)
        
    def close_pool(self):
        self.pool.close()
        
    def setup_proposal(self, **kwargs):
        
        # initialise the kernel.
        self.link_factory = kwargs['link_factory']
        
        # pass the kwargs to the kernel.
        self.kernel.setup_proposal(**kwargs)
        
    def adapt(self, **kwargs):
        
        # this method is not adaptive in its own, but its kernel might be.
        self.kernel.adapt(**kwargs)
        
    def make_proposal(self, link):
        
        # create proposals. this is fast so no paralellised.
        proposals = [self.kernel.make_proposal(link) for i in range(self.k)]
        
        # get the links in parallel.
        self.proposal_links = self.pool.map(self.link_factory.create_link, proposals)
        
        # get the unnormalised weights according to the proposal type.
        if isinstance(self.kernel, CrankNicolson):
            self.proposal_weights = np.array([link.likelihood for link in self.proposal_links])
        elif isinstance(self.kernel, GaussianRandomWalk):
            self.proposal_weights = np.array([link.posterior for link in self.proposal_links])
        
        # unless all proposals are extremely unlikely, return one link according to the density.
        if not np.isinf(self.proposal_weights).all():
            return np.random.choice(self.proposal_links, p=np.exp(self.proposal_weights - logsumexp(self.proposal_weights))).parameters
        # otherwise, just return a link at random (will be rejected anyway)
        else:
            return np.random.choice(self.proposal_links).parameters
    
    def get_acceptance(self, proposal_link, previous_link):
        
        # check if the proposal makes sense, if not return 0.
        if np.isnan(proposal_link.posterior) or np.isinf(self.proposal_weights).all():
            return 0
        
        else:
            
            # create reference proposals.this is fast so no paralellised.
            references = [self.kernel.make_proposal(proposal_link) for i in range(self.k-1)]
            
            # get the links in parallel.
            self.reference_links = self.pool.map(self.link_factory.create_link, references)
            
            # get the unnormalised weights according to the proposal type.
            if isinstance(self.kernel, CrankNicolson):
                self.reference_weights = np.array([link.likelihood for link in self.reference_links] + [previous_link.likelihood])
            elif isinstance(self.kernel, GaussianRandomWalk):
                self.reference_weights = np.array([link.posterior for link in self.reference_links] + [previous_link.posterior])
            
            # get the acceptance probability.
            return np.exp(logsumexp(self.proposal_weights) - logsumexp(self.reference_weights))
