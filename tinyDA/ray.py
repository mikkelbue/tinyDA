import ray
import warnings

from itertools import compress
import numpy as np
from tqdm import tqdm
from scipy.special import logsumexp

from .chain import Chain, DAChain
from .proposal import *
from .utils import *


class ParallelChain:
    '''
    ParallelChain crates n_chains instances of tda.Chain and runs the chains in parallel.
    '''
    def __init__(self, link_factory, proposal, n_chains=2, initial_parameters=None):
        
        # do not use MultipleTry proposal with ParallelChain, since that will create nested 
        # instances in Ray, which will be competing for resources. This can be very slow.
        # TODO: create better Ray implementation, that circumvents this.
        if isinstance(proposal, MultipleTry):
            raise TypeError('MultipleTry proposal cannot be used with ParallelChain')
        
        # if the proposal is pCN, Check if the proposal covariance is equal 
        # to the prior covariance and if the prior is zero mean.
        elif isinstance(proposal, CrankNicolson) and not isinstance(link_factory.prior, stats._multivariate.multivariate_normal_frozen):
            raise TypeError('Prior must be of type scipy.stats.multivariate_normal for pCN proposal')
        
        # internalise the link factory and proposal.
        self.link_factory = link_factory
        self.proposal = proposal
        
        # set the number of parallel chains and initial parameters.
        self.n_chains = n_chains
        self.initial_parameters = initial_parameters
        
        # check if the given initial parameters are the right type and size.
        if self.initial_parameters is not None:
            if type(self.initial_parameters) == list:
                assert len(self.initial_parameters) == self.n_chains, 'If list of initial parameters is provided, it must have length n_chains'
            else:
                raise TypeError('Initial parameters must be a list')
        # if no initial parameters were given, generate some from the prior.
        else:
            self.initial_parameters = list(self.link_factory.prior.rvs(self.n_chains))
        
        # initialise Ray.
        ray.init(ignore_reinit_error=True)
        
        # set up the parallel chains as Ray actors.
        self.remote_chains = [RemoteChain.remote(self.link_factory, 
                                                 self.proposal, 
                                                 initial_parameters) for initial_parameters in self.initial_parameters]
        
    def sample(self, iterations, progressbar=True):
        
        # initialise sampling on the chains and fetch the results.
        processes = [chain.sample.remote(iterations) for chain in self.remote_chains]
        results = ray.get(processes)
        self.chains = [result['chain'] for result in results]
        self.accepted = [result['accepted'] for result in results]
        

class ParallelDAChain(ParallelChain):
    '''
    ParalleDAChain crates n_chains instances of tda.DAChain and runs the chains in parallel.
    '''
    def __init__(self, link_factory_coarse, link_factory_fine, proposal, subsampling_rate=1, n_chains=2, initial_parameters=None, adaptive_error_model=None, R=None):
        
        # do not use MultipleTry proposal with ParallelDAChain, since that will create nested 
        # instances in Ray, which will be competing for resources. This can be very slow.
        # TODO: create better Ray implementation, that circumvents this.
        if isinstance(proposal, MultipleTry):
            raise TypeError('MultipleTry proposal cannot be used with ParallelDAChain')
        
        # if the proposal is pCN, Check if the proposal covariance is equal 
        # to the prior covariance and if the prior is zero mean.
        elif isinstance(proposal, CrankNicolson) and not isinstance(link_factory_coarse.prior, stats._multivariate.multivariate_normal_frozen):
            raise TypeError('Prior must be of type scipy.stats.multivariate_normal for pCN proposal')
        
        # internalise link factories, proposal and subsampling rate.
        self.link_factory_coarse = link_factory_coarse
        self.link_factory_fine = link_factory_fine
        self.proposal = proposal
        self.subsampling_rate = subsampling_rate
        
        # set the number of parallel chains and initial parameters.
        self.n_chains = n_chains
        self.initial_parameters = initial_parameters
        
        # check if the given initial parameters are the right type and size.
        if self.initial_parameters is not None:
            if type(self.initial_parameters) == list:
                assert len(self.initial_parameters) == self.n_chains, 'If list of initial parameters is provided, it must have length n_chains'
            else:
                raise TypeError('Initial parameters must be a list')
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
                                                   
class PopulationChain:
    def __init__(self, link_factory, proposal, n_chains=2, initial_parameters=None):
        
        # do not use MultipleTry proposal with ParallelChain, since that will create nested 
        # instances in Ray, which will be competing for resources. This can be very slow.
        # TODO: create better Ray implementation, that circumvents this.
        if isinstance(proposal, MultipleTry):
            raise TypeError('MultipleTry proposal cannot be used with PopulationChain')
        
        # if the proposal is pCN, Check if the proposal covariance is equal 
        # to the prior covariance and if the prior is zero mean.
        elif isinstance(proposal, CrankNicolson) and not isinstance(link_factory.prior, stats._multivariate.multivariate_normal_frozen):
            raise TypeError('Prior must be of type scipy.stats.multivariate_normal for pCN proposal')
        
        # internalise the link factory and proposal.
        self.link_factory = link_factory
        self.proposal = proposal
        
        # set the number of parallel chains and initial parameters.
        self.n_chains = n_chains
        
        # initialise a list, which holds the links.
        self.chains = [[] for i in range(self.n_chains)]
        
        # initialise a list, which holds boolean acceptance values.
        self.accepted = [[] for i in range(self.n_chains)]
        
        self.initial_parameters = initial_parameters
        
        # check if the given initial parameters are the right type and size.
        if self.initial_parameters is not None:
            if type(self.initial_parameters) == list:
                assert len(self.initial_parameters) == self.n_chains, 'If list of initial parameters is provided, it must have length n_chains'
            else:
                raise TypeError('Initial parameters must be a list')
        # if no initial parameters were given, generate some from the prior.
        else:
            self.initial_parameters = list(self.link_factory.prior.rvs(self.n_chains))
        
        # initialise Ray.
        ray.init(ignore_reinit_error=True)
        
        # set up the parallel chains as Ray actors.
        self.link_factories = [RemoteLinkFactory.remote(self.link_factory) for i in range(self.n_chains)]
        
        # append a link with the initial parameters (this will automatically
        # compute the model output, and the relevant probabilties.
        processes = [link_factory.create_link.remote(initial_parameters) for link_factory, initial_parameters in zip(self.link_factories, self.initial_parameters)]
        [chain.append(ray.get(process)) for chain, process in zip(self.chains, processes)]
        [accept.append(True) for accept in self.accepted]
        
        # setup the proposal
        self.proposal.setup_proposal(parameters=self.initial_parameters[0], link_factory=self.link_factory)
        
    def sample(self, iterations, progressbar=True):
        
        # start the iteration
        if progressbar:
            pbar = tqdm(range(iterations))
        else:
            pbar = range(iterations)
        
        for i in pbar:
            
            if progressbar:
                
                acceptance = ['{:.2f}'.format(np.mean(self.accepted[i][-100:])) for i in range(self.n_chains)]
                pbar.set_description('Running chains, \u03B1 = {}'.format(acceptance))
            
            # draw a new proposal, given the previous parameters.
            proposals = [self.proposal.make_proposal(chain[-1]) for chain in self.chains]
            
            # create a link from that proposal
            processes = [link_factory.create_link.remote(proposal) for link_factory, proposal in zip(self.link_factories, proposals)]
            
            for i in range(self.n_chains):
                
                proposal_link = ray.get(processes[i])
            
                # compute the acceptance probability, which is unique to
                # the proposal.
                alpha = self.proposal.get_acceptance(proposal_link, self.chains[i][-1])
            
                # perform Metropolis adjustment.
                if np.random.random() < alpha:
                    self.chains[i].append(proposal_link)
                    self.accepted[i].append(True)
                else:
                    self.chains[i].append(self.chains[i][-1])
                    self.accepted[i].append(False)
            
                # adapt the proposal. if the proposal is set to non-adaptive,
                # this has no effect.
                self.proposal.adapt(parameters=self.chains[i][-1].parameters, 
                                    jumping_distance=self.chains[i][-1].parameters-self.chains[i][-2].parameters, 
                                    accepted=self.accepted[i])
        if progressbar:
            pbar.close()

class FetchingDAChain:
    def __init__(self, link_factory_coarse, link_factory_fine, proposal, subsampling_rate=1, fetching_rate=1, initial_parameters=None):
        
        # do not use MultipleTry proposal with FetchingDAChain, since that will create nested 
        # instances in Ray, which will be competing for resources. This can be very slow.
        # TODO: create better Ray implementation, that circumvents this.
        if isinstance(proposal, MultipleTry):
            raise TypeError('MultipleTry proposal cannot be used with FetchingDAChain')
        
        # if the proposal is pCN, Check if the proposal covariance is equal 
        # to the prior covariance and if the prior is zero mean.
        elif isinstance(proposal, CrankNicolson) and not isinstance(link_factory_coarse.prior, stats._multivariate.multivariate_normal_frozen):
            raise TypeError('Prior must be of type scipy.stats.multivariate_normal for pCN proposal')
        
        # internalise link factories proposal, subsampling rate and fetching rate.
        self.link_factory_coarse = link_factory_coarse
        self.link_factory_fine = link_factory_fine
        self.proposal = proposal
        self.subsampling_rate = subsampling_rate
        self.fetching_rate = fetching_rate
        
        # set up lists to hold coarse and fine links, as well as acceptance accounting.
        self.chain_coarse = []
        self.accepted_coarse = []
        self.is_coarse = []
        
        self.chain_fine = []
        self.accepted_fine = []
        self.perfect_fetch = []
                
        # if the initial parameters are given, use them. otherwise,
        # draw a random sample from the prior.
        if initial_parameters is not None:
            self.initial_parameters = initial_parameters
        else:
            self.initial_parameters = self.link_factory_coarse.prior.rvs()
            
        # setup the proposal
        self.proposal.setup_proposal(parameters=self.initial_parameters, link_factory=self.link_factory_coarse)
        
        # initialise Ray.
        ray.init(ignore_reinit_error=True)
        
        # set up the coarse and fine workers.
        self.coarse_workers = [RemoteSubchainFactory.remote(self.link_factory_coarse, self.subsampling_rate, self.fetching_rate) for i in range(self.fetching_rate+1)]
        self.fine_workers = [RemoteLinkFactory.remote(self.link_factory_fine) for i in range(self.fetching_rate)]
        
        # create an initial coarse link.
        initial_coarse_link = self.link_factory_coarse.create_link(self.initial_parameters)
        
        # initialise a coarse subchain and a find link from the initial parameters.
        coarse_process = self.coarse_workers[0].run.remote(initial_coarse_link, self.proposal)
        fine_process = self.fine_workers[0].create_link.remote(self.initial_parameters)
        
        # get the initial coarse section, and fine link.
        self.coarse_section = ray.get(coarse_process)
        self.chain_fine.append(ray.get(fine_process))
        self.accepted_fine.append(True)
        
    def sample(self, iterations, progressbar=True):
        
        # start the iteration
        if progressbar:
            pbar = tqdm(total=iterations)
        
        # the true number of fetching iterations is unknown, so we run until the fine chain is long enough.
        while True:
            
            # update the progressbar
            if progressbar:
                pbar.set_description('Running chain, \u03B1_c = {0:.3f}, \u03B1_f = {1:.2f}'.format(np.mean(self.accepted_coarse[-int(100*self.subsampling_rate):]), np.mean(self.accepted_fine[-100:])))
            
            # get all the coarse links that are proper proposals.
            nodes = [link for link, is_coarse in zip(self.coarse_section['chain'], self.coarse_section['is_coarse']) if not is_coarse]
            
            # initialise coarse subchains and fine evaluations.
            coarse_processes = [coarse_worker.run.remote(node, self.proposal) for coarse_worker, node in zip(self.coarse_workers, nodes)]
            fine_processes = [fine_worker.create_link.remote(node.parameters) for fine_worker, node in zip(self.fine_workers, nodes[1:])]
            
            # get the results.
            coarse_sections = ray.get(coarse_processes); fine_links = ray.get(fine_processes)
            
            # set the perfect fetch switch.
            self.perfect_fetch.append(True)
            
            # iterate through the fine links.
            for i in range(self.fetching_rate):
                
                # perform Metropolis adjustment..
                alpha = np.exp(fine_links[i].posterior - self.chain_fine[-1].posterior + nodes[i].posterior - nodes[i+1].posterior)
                 
                if np.random.random() < alpha:
                    self.chain_fine.append(fine_links[i])
                    self.accepted_fine.append(True)
                else:
                    self.chain_fine.append(self.chain_fine[-1])
                    self.accepted_fine.append(False)
                    self.perfect_fetch[-1] = False
                    break
                    
            fine_length = i + 1
            
            # extend the coarse chain according to the accepted fine links
            coarse_length = fine_length*(self.subsampling_rate+1)
            self.chain_coarse.extend(self.coarse_section['chain'][:coarse_length])
            self.accepted_coarse.extend(self.coarse_section['accepted'][:coarse_length])
            self.is_coarse.extend(self.coarse_section['is_coarse'][:coarse_length])
            
            # get the correct section for next iteration.
            if self.perfect_fetch[-1]:
                self.coarse_section = coarse_sections[fine_length]
            else:
                self.coarse_section = coarse_sections[fine_length-1]
            
            # adapt the proposal using only the "proper" coarse links.
            for i in range(coarse_length):
                if self.is_coarse[-coarse_length+i]:
                    self.proposal.adapt(parameters=self.chain_coarse[-coarse_length+i].parameters, 
                                        jumping_distance=self.chain_coarse[-coarse_length+i].parameters-self.chain_coarse[-coarse_length+i-1].parameters, 
                                        accepted=list(compress(self.accepted_coarse[:-coarse_length+i], self.is_coarse[:-coarse_length+i])))
            
            # update the progress bar.
            if progressbar:    
                pbar.update(fine_length)
            
            # break if the fine chain is long enough.
            if len(self.chain_fine) >= iterations:
                break
        
        # close the progress bar.
        if progressbar:
            pbar.close()

class MultipleTry(Proposal):
    
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
class RemoteChain(Chain):
    def sample(self, iterations, progressbar=False):
        super().sample(iterations, progressbar)
        
        return {'chain': self.chain, 'accepted': self.accepted}
        
@ray.remote
class RemoteDAChain(DAChain):
    def sample(self, iterations, progressbar=False):
        super().sample(iterations, progressbar)
        
        return {'chain': self.chain_fine, 'accepted': self.accepted_fine}

@ray.remote
class RemoteLinkFactory:
    def __init__(self, link_factory):
        self.link_factory = link_factory
        
    def create_link(self, parameters):
        return self.link_factory.create_link(parameters)

@ray.remote
class RemoteSubchainFactory:
    
    def __init__(self, link_factory, subsampling_rate, fetching_rate):
        
        self.link_factory = link_factory
        self.subsampling_rate = subsampling_rate
        self.fetching_rate = fetching_rate
        
    def run(self, initial_link, proposal_kernel):
        
        chain = [initial_link]
        accepted = [True]
        is_coarse = [False]
        
        for i in range(self.fetching_rate):
        
            for j in range(self.subsampling_rate):
                
                # draw a new proposal, given the previous parameters.
                proposal = proposal_kernel.make_proposal(chain[-1])
                
                # create a link from that proposal.
                proposal_link = self.link_factory.create_link(proposal)
                
                # compute the acceptance probability, which is unique to
                # the proposal.
                alpha = proposal_kernel.get_acceptance(proposal_link, chain[-1])
                
                # perform Metropolis adjustment.
                if np.random.random() < alpha:
                    chain.append(proposal_link)
                    accepted.append(True)
                    is_coarse.append(True)
                else:
                    chain.append(chain[-1])
                    accepted.append(False)
                    is_coarse.append(True)
        
            chain.append(chain[-1])
            accepted.append(True)
            is_coarse.append(False)
            
        return {'chain': chain, 'accepted': accepted, 'is_coarse': is_coarse}

