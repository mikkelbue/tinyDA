import ray
import warnings

from itertools import compress
import numpy as np
from scipy.special import logsumexp

from .chain import Chain, DAChain
from .proposal import *

class ParallelChain:

    '''
    ParallelChain creates n_chains instances of tinyDA.Chain and runs the chains in parallel.
    It is initialsed with a LinkFactory (which holds the model and the distributions, and
    returns Links), and a proposal (transition kernel).

    Attributes
    ----------
    link_factory : tinyDA.LinkFactory
        A link factory responsible for communation between prior, likelihood and model.
        It also generates instances of tinyDA.Link (sample objects).
    proposal : tinyDA.Proposal
        Transition kernel for MCMC proposals.
    n_chains : int
        Number of parallel chains.
    initial_parameters : list
        Starting points for the MCMC samplers
    remote_chains : list
        List of Ray actors, each running an independent MCMC sampler.
    chains : list
        List of lists containing samples ("Links") in the MCMC chains.
    accepted : list
        List of lists of bool, signifying whether a proposal was accepted or not.

    Methods
    -------
    sample(iterations)
        Runs the MCMC for the specified number of iterations.
    '''

    def __init__(self, link_factory, proposal, n_chains=2, initial_parameters=None):

        '''
        Parameters
        ----------
        link_factory : tinyDA.LinkFactory
            A link factory responsible for communation between prior, likelihood and model.
            It also generates instances of tinyDA.Link (sample objects).
        proposal : tinyDA.Proposal
            Transition kernel for MCMC proposals.
        n_chains : int, optional
            Number of independent MCMC samplers. Default is 2.
        initial_parameters : list, optional
            Starting points for the MCMC samplers, default is None (random draws from prior).
        '''

        # internalise the link factory and proposal.
        self.link_factory = link_factory
        self.proposal = proposal

        # set the number of parallel chains and initial parameters.
        self.n_chains = n_chains

        # set the initial parameters.
        if initial_parameters is not None:
            self.initial_parameters = initial_parameters
        # if no initial parameters were given, generate some from the prior.
        else:
            self.initial_parameters = list(self.link_factory.prior.rvs(self.n_chains))

        # initialise Ray.
        ray.init(ignore_reinit_error=True)

        # set up the parallel chains as Ray actors.
        self.remote_chains = [RemoteChain.remote(self.link_factory,
                                                 self.proposal,
                                                 initial_parameters) for initial_parameters in self.initial_parameters]

    def sample(self, iterations, progressbar=False):

        '''
        Parameters
        ----------
        iterations : int
            Number of MCMC samples to generate.
        progressbar : bool, optional
            Whether to draw a progressbar, default is False, since Ray
            and tqdm do not play very well together.
        '''

        # initialise sampling on the chains and fetch the results.
        processes = [chain.sample.remote(iterations, progressbar) for chain in self.remote_chains]
        self.chains = ray.get(processes)


class ParallelDAChain(ParallelChain):

    '''
    ParalleDAChain creates n_chains instances of tinyDA.DAChain and runs the
    chains in parallel. It takes a coarse and a fine link factory as input,
    as well as a proposal, which applies to the coarse level only.

    Attributes
    ----------
    link_factory_coarse : tinyDA.LinkFactory
        A "coarse" link factory responsible for communation between prior, likelihood and model.
        It also generates instances of tinyDA.Link (sample objects).
    link_factory_fine : tinyDA.LinkFactory
        A "fine" link factory responsible for communation between prior, likelihood and model.
        It also generates instances of tinyDA.Link (sample objects).
    proposal : tinyDA.Proposal
        Transition kernel for coarse MCMC proposals.
    subsampling_rate : int
        The subsampling rate for the coarse chain.
    n_chains : int
        Number of parallel chains.
    initial_parameters : list
            Starting points for the MCMC samplers.
    adaptive_error_model : str or None
        The adaptive error model, see e.g. Cui et al. (2019).
    R : numpy.ndarray
        Restriction matrix for the adaptive error model.
    remote_chains : list
        List of Ray actors, each running an independent DA MCMC sampler.
    chains : list
        List of lists containing samples ("Links") in the fine MCMC chains.
    accepted : list
        List of lists of bool, signifying whether a proposal was accepted or not.

    Methods
    -------
    sample(iterations)
        Runs the MCMC for the specified number of iterations.
    '''

    def __init__(self, link_factory_coarse, link_factory_fine, proposal, subsampling_rate=1, n_chains=2, initial_parameters=None, adaptive_error_model=None, R=None):

        '''
        Parameters
        ----------
        link_factory_coarse : tinyDA.LinkFactory
            A "coarse" link factory responsible for communation between prior, likelihood and model.
            It also generates instances of tinyDA.Link (sample objects).
        link_factory_fine : tinyDA.LinkFactory
            A "fine" link factory responsible for communation between prior, likelihood and model.
            It also generates instances of tinyDA.Link (sample objects).
        proposal : tinyDA.Proposal
            Transition kernel for coarse MCMC proposals.
        subsampling_rate : int, optional
            The subsampling rate for the coarse chain. The default is 1, resulting in "classic" DA MCMC..
        n_chains : int, optional
            Number of independent MCMC samplers. Default is 2.
        initial_parameters : list, optional
            Starting points for the MCMC samplers, default is None (random draws from prior).
        adaptive_error_model : str or None, optional
            The adaptive error model, see e.g. Cui et al. (2019). Default is None (no error model),
            options are 'state-independent' or 'state-dependent'. If an error model is used, the
            likelihood MUST have a set_bias() method, use e.g. tinyDA.AdaptiveLogLike.
        R : numpy.ndarray, optional
            Restriction matrix for the adaptive error model. Default is None (identity matrix).
        '''

        # internalise link factories, proposal and subsampling rate.
        self.link_factory_coarse = link_factory_coarse
        self.link_factory_fine = link_factory_fine
        self.proposal = proposal
        self.subsampling_rate = subsampling_rate

        # set the number of parallel chains and initial parameters.
        self.n_chains = n_chains

        # set the initial parameters.
        if initial_parameters is not None:
            self.initial_parameters = initial_parameters
        # if no initial parameters were given, generate some from the prior.
        else:
            self.initial_parameters = list(self.link_factory_coarse.prior.rvs(self.n_chains))

        # set the adaptive error model
        self.adaptive_error_model = adaptive_error_model
        self.R = R

        # initialise Ray.
        ray.init(ignore_reinit_error=True)

         # set up the parallel DA chains as Ray actors.
        self.remote_chains = [RemoteDAChain.remote(self.link_factory_coarse,
                                                   self.link_factory_fine,
                                                   self.proposal,
                                                   self.subsampling_rate,
                                                   initial_parameters,
                                                   self.adaptive_error_model, self.R) for initial_parameters in self.initial_parameters]

@ray.remote
class RemoteChain(Chain):
    def sample(self, iterations, progressbar):
        super().sample(iterations, progressbar)

        return self.chain

@ray.remote
class RemoteDAChain(DAChain):
    def sample(self, iterations, progressbar):
        super().sample(iterations, progressbar)

        return list(compress(self.chain_coarse, self.is_coarse)), self.chain_fine

class MultipleTry(Proposal):
    
    '''
    Multiple-Try proposal (Liu et al. 2000), which will take any other 
    TinyDA proposal as a kernel. If the kernel is symmetric, it uses MTM(II), 
    otherwise it uses MTM(I). The parameter k sets the number of tries.
    
    Attributes
    ----------
    kernel : tinyDA.Proposal
        The kernel of the Multiple-Try proposal (another proposal).
    k : int
        Number of mutiple tries.
        
    Methods
    ----------
    setup_proposal(**kwargs)
        Initialises the kernel, and the remote LinkFactories.
    adapt(**kwargs)
        Adapts the kernel.
    make_proposal(link)
        Generates a Multiple Try proposal, using the kernel.
    get_acceptance(proposal_link, previous_link)
        Computes the acceptance probability given a proposal link and the previous link.
    '''
    
    is_symmetric = True
    
    def __init__(self, kernel, k):
        
        '''
        Parameters
        ----------
        kernel : tinyDA.Proposal
            The kernel of the Multiple-Try proposal (another proposal)
        k : int
            Number of mutiple tries.
        '''
        
        # set the kernel
        self.kernel = kernel
        
        # set the number of tries per proposal.
        self.k = k
        
        if self.kernel.adaptive:
            warnings.warn(' Using global adaptive scaling with MultipleTry proposal can be unstable.\n')
            
        ray.init(ignore_reinit_error=True)
        
    def setup_proposal(self, **kwargs):
        
        # pass the kwargs to the kernel.
        self.kernel.setup_proposal(**kwargs)
        
        # initialise the link factories.
        self.link_factories = [RemoteLinkFactory.remote(kwargs['link_factory']) for i in range(self.k)]
        
    def adapt(self, **kwargs):
        
        # this method is not adaptive in its own, but its kernel might be.
        self.kernel.adapt(**kwargs)
        
    def make_proposal(self, link):
        
        # create proposals. this is fast so no paralellised.
        proposals = [self.kernel.make_proposal(link) for i in range(self.k)]
        
        # get the links in parallel.
        proposal_processes = [link_factory.create_link.remote(proposal) for proposal, link_factory in zip(proposals, self.link_factories)]
        self.proposal_links = ray.get(proposal_processes)
        
        # if kernel is symmetric, use MTM(II), otherwise use MTM(I).
        if self.kernel.is_symmetric:
            q_x_y = np.zeros(self.k)
        else:
            q_x_y = np.array([self.kernel.get_q(link, proposal_link) for proposal_link in self.proposal_links])
        
        # get the unnormalised weights.
        self.proposal_weights = np.array([link.posterior+q for link, q in zip(self.proposal_links, q_x_y)])
        self.proposal_weights[np.isnan(self.proposal_weights)] = -np.inf 
        
        # if all posteriors are -Inf, return a random one.
        if np.isinf(self.proposal_weights).all():
            return np.random.choice(self.proposal_links).parameters
        
        # otherwise, return a random one according to the weights.
        else:            
            return np.random.choice(self.proposal_links, p=np.exp(self.proposal_weights - logsumexp(self.proposal_weights))).parameters
    
    def get_acceptance(self, proposal_link, previous_link):
        
        # check if the proposal makes sense, if not return 0.
        if np.isnan(proposal_link.posterior) or np.isinf(self.proposal_weights).all():
            return 0
        
        else:
            
            # create reference proposals.this is fast so no paralellised.
            references = [self.kernel.make_proposal(proposal_link) for i in range(self.k-1)]
            
            # get the links in parallel.
            reference_processes = [link_factory.create_link.remote(reference) for reference, link_factory in zip(references, self.link_factories)]
            self.reference_links = ray.get(reference_processes)
            
            # if kernel is symmetric, use MTM(II), otherwise use MTM(I).
            if self.kernel.is_symmetric:
                q_y_x = np.zeros(self.k)
            else:
                q_y_x = np.array([self.kernel.get_q(proposal_link, reference_link) for reference_link in self.reference_links])
            
            # get the unnormalised weights.
            self.reference_weights = np.array([link.posterior+q for link, q in zip(self.reference_links, q_y_x)])
            self.reference_weights[np.isnan(self.reference_weights)] = -np.inf 
            
            # get the acceptance probability.
            return np.exp(logsumexp(self.proposal_weights) - logsumexp(self.reference_weights))

@ray.remote
class RemoteLinkFactory:
    def __init__(self, link_factory):
        self.link_factory = link_factory
        
    def create_link(self, parameters):
        return self.link_factory.create_link(parameters)

