import ray

import warnings
import numpy as np
from scipy.special import logsumexp

from .chain import Chain, DAChain

class ParallelChain:
    def __init__(self, link_factory, proposal, n_chains=2, initial_parameters=None):
        
        self.link_factory = link_factory
        self.proposal = proposal
        
        self.n_chains = n_chains
        self.initial_parameters = initial_parameters
        
        if self.initial_parameters is not None:
            if type(self.initial_parameters) == list:
                assert len(self.initial_parameters) == self.n_chains, 'If list of initial parameters is provided, it must have length n_chains'
            else:
                raise TypeError('Initial parameters must be a list')
        else:
            self.initial_parameters = list(self.link_factory.prior.rvs(self.n_chains))
        
        ray.init(ignore_reinit_error=True)
        
        self.remote_chains = [RemoteChain.remote(self.link_factory, 
                                                 self.proposal, 
                                                 initial_parameters) for initial_parameters in self.initial_parameters]
        
    def sample(self, iterations, progressbar=True):
        
        self.processes = [chain.sample.remote(iterations) for chain in self.remote_chains]
        self.chains = [ray.get(process) for process in self.processes]
        
class ParallelDAChain(ParallelChain):
    def __init__(self, link_factory_coarse, link_factory_fine, proposal, subsampling_rate=1, n_chains=2, initial_parameters=None, adaptive_error_model=None, R=None):
        
        # internalise link factories and the proposal
        self.link_factory_coarse = link_factory_coarse
        self.link_factory_fine = link_factory_fine
        self.proposal = proposal
        self.subsampling_rate = subsampling_rate
        
        self.n_chains = n_chains
        self.initial_parameters = initial_parameters
        
        self.adaptive_error_model = adaptive_error_model
        self.R = R
        
        if self.initial_parameters is not None:
            if type(self.initial_parameters) == list:
                assert len(self.initial_parameters) == self.n_chains, 'If list of initial parameters is provided, it must have length n_chains'
            else:
                raise TypeError('Initial parameters must be a list')
        else:
            self.initial_parameters = list(self.link_factory_coarse.prior.rvs(self.n_chains))
        
        ray.init(ignore_reinit_error=True)
        
        self.remote_chains = [RemoteDAChain.remote(self.link_factory_coarse, 
                                                   self.link_factory_fine, 
                                                   self.proposal, 
                                                   self.subsampling_rate, 
                                                   initial_parameters, 
                                                   self.adaptive_error_model, self.R) for initial_parameters in self.initial_parameters]
        

@ray.remote
class RemoteChain(Chain):
    def sample(self, iterations, progressbar=False):
        super().sample(iterations, progressbar)
        
        return self.chain
        
@ray.remote
class RemoteDAChain(DAChain):
    def sample(self, iterations, progressbar=False):
        super().sample(iterations, progressbar)
        
        return self.chain_fine


class MultipleTry:
    
    '''
    Multiple-Try proposal, which will take any other TinyDA proposal
    as a kernel. The parameter k sets the number of tries.
    '''
    
    is_symmetric = True
    
    def __init__(self, kernel, k):
        
        ray.init(ignore_reinit_error=True)
        
        # set the kernel
        self.kernel = kernel
        
        # set the number of tries per proposal.
        self.k = k
        
        if self.kernel.adaptive:
            warnings.warn(' Using global adaptive scaling with MultipleTry proposal can be unstable.\n')
        
    def setup_proposal(self, **kwargs):
        
        # pass the kwargs to the kernel.
        self.kernel.setup_proposal(**kwargs)
        
        # initialise the link factories.
        self.link_factories = [LinkFactoryWrapper.remote(kwargs['link_factory']) for i in range(self.k)]
        
    def adapt(self, **kwargs):
        
        # this method is not adaptive in its own, but its kernel might be.
        self.kernel.adapt(**kwargs)
        
    def make_proposal(self, link):
        
        # create proposals. this is fast so no paralellised.
        proposals = [self.kernel.make_proposal(link) for i in range(self.k)]
        
        # get the links in parallel.
        proposal_processes = [link_factory.create_link.remote(proposal) for proposal, link_factory in zip(proposals, self.link_factories)]
        self.proposal_links = [ray.get(proposal_process) for proposal_process in proposal_processes]
        
        # if kernel is symmetric, use MTM(II), otherwise use MTM(I).
        if self.kernel.is_symmetric:
            q_x_y = np.zeros(self.k)
        else:
            q_x_y = np.array([self.kernel.get_q(link, proposal_link) for proposal_link in self.proposal_links])
        
        # get the unnormalised weights.
        self.proposal_weights = np.array([link.posterior+q for link, q in zip(self.proposal_links, q_x_y)])
        self.proposal_weights[np.isnan(self.proposal_weights)] = -np.inf 
        
        # if all posteriors are -Inf, return a random onw.
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
            self.reference_links = [ray.get(reference_process) for reference_process in reference_processes]
            
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
class LinkFactoryWrapper:
    def __init__(self, link_factory):
        self.link_factory = link_factory
        
    def create_link(self, parameters):
        return self.link_factory.create_link(parameters)
