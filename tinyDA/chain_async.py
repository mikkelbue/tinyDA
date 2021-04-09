# externa; imports
import ray
from itertools import compress
import numpy as np
from tqdm import tqdm

# internal imports
from .proposal import *
from .utils import *
            
class ADAChain:
    
    '''
    DAChain is a two-level Delayed Acceptance sampler with finite length
    subchains. It takes a coarse and a fine link factory as input, as well
    as a proposal, which applies to the coarse level only.
    The adaptive_error_model can be set to either 'state-independent' or
    'state-dependent'. If the adaptive_error_model is used, the likelihood
    must have a set_bias() method. See example in the distributions.py.
    '''
    
    def __init__(self, link_factory_coarse, link_factory_fine, proposal, subsampling_rate=1, initial_parameters=None, adaptive_error_model=None, R=None):
        
        # internalise link factories and the proposal
        self.link_factory_coarse = link_factory_coarse
        self.link_factory_fine = link_factory_fine
        self.proposal = proposal
        self.subsampling_rate = subsampling_rate
        
        # if the proposal is pCN, Check if the proposal covariance is equal 
        # to the prior covariance and if the prior is zero mean.
        if isinstance(self.proposal, CrankNicolson):
            if isinstance(self.link_factory_coarse.prior, stats._multivariate.multivariate_normal_frozen):
                if not (self.link_factory_coarse.prior.cov == self.proposal.C).all():
                    raise ValueError('C-proposal must equal C-prior for pCN proposal')
                if np.count_nonzero(self.link_factory_coarse.prior.mean):
                    raise ValueError('Prior must be zero mean for pCN proposal')
            else:
                raise TypeError('Prior must be of type scipy.stats.multivariate_normal for pCN proposal')
        
        # check the same if the CrankNicolson is nested in a MultipleTry proposal.
        elif isinstance(self.proposal, MultipleTry):
            if isinstance(self.proposal.kernel, CrankNicolson):
                if isinstance(self.link_factory_coarse.prior, stats._multivariate.multivariate_normal_frozen):
                    if not (self.link_factory_coarse.prior.cov == self.proposal.kernel.C).all():
                        raise ValueError('C-proposal must equal C-prior for pCN kernel')
                    if np.count_nonzero(self.link_factory_coarse.prior.mean):
                        raise ValueError('Prior must be zero mean for pCN kernel')
                else:
                    raise TypeError('Prior must be of type scipy.stats.multivariate_normal for pCN kernel')
                
        # set up lists to hold coarse and fine links, as well as acceptance
        # accounting
        self.chain_coarse = []
        self.accepted_coarse = []
        self.is_coarse = []
        
        self.chain_fine = []
        self.accepted_fine = []
                
        # if the initial parameters are given, use them. otherwise,
        # draw a random sample from the prior.
        if initial_parameters is not None:
            self.initial_parameters = initial_parameters
        else:
            self.initial_parameters = self.link_factory_coarse.prior.rvs()
            
        # create initial coarse link..
        initial_coarse_link = self.link_factory_coarse.create_link(self.initial_parameters)
        
        # append a link with the initial parameters to the coarse chain.
        self.chain_fine.append(self.link_factory_fine.create_link(self.initial_parameters))
        self.accepted_fine.append(True)
        
        # if the proposal is AM, use the initial parameters as the first
        # sample for the RecursiveSamplingMoments.
        if isinstance(self.proposal, AdaptiveMetropolis) or isinstance(self.proposal, AdaptiveCrankNicolson):
            self.proposal.initialise_sampling_moments(self.initial_parameters)
            
        elif isinstance(self.proposal, SingleDreamZ):
            self.proposal.initialise_archive(self.link_factory_coarse.prior)
            
        elif isinstance(self.proposal, MultipleTry):
            self.proposal.initialise_kernel(self.link_factory_coarse, self.initial_parameters)
        
        # set the adative error model as a. attribute.
        self.adaptive_error_model = adaptive_error_model
        
        # set up the adaptive error model.
        if self.adaptive_error_model is not None:
            
            if R is None:
                self.R = np.eye(self.chain_fine[-1].model_output.shape[0])
            else:
                self.R = R
            
            # compute the difference between coarse and fine level.
            self.model_diff = self.R.dot(self.chain_fine[-1].model_output) - initial_coarse_link.model_output
            
            if self.adaptive_error_model == 'state-independent':
                # for the state-independent error model, the bias is 
                # RecursiveSampleMoments, and the corrector is the mean 
                # of all sampled differences.
                self.bias = RecursiveSampleMoments(self.model_diff, np.zeros((self.model_diff.shape[0], self.model_diff.shape[0])))
                self.link_factory_coarse.likelihood.set_bias(self.bias.get_mu(), self.bias.get_sigma())
            
            elif self.adaptive_error_model == 'state-dependent':
                # for the state-dependent error model, the bias is 
                # ZeroMeanRecursiveSampleMoments, and the corrector is 
                # the last sampled difference.
                self.bias = ZeroMeanRecursiveSampleMoments(np.zeros((self.model_diff.shape[0], self.model_diff.shape[0])))
                self.link_factory_coarse.likelihood.set_bias(self.model_diff, self.bias.get_sigma())
        
            initial_coarse_link = self.link_factory_coarse.update_link(initial_coarse_link)
            
        ray.init(ignore_reinit_error=True)
        
        if isinstance(self.proposal, MultipleTry):
            self.proposal.open_pool()
        
        proposal_process = coarse_worker.remote(initial_coarse_link, self.link_factory_coarse, self.proposal, self.subsampling_rate, True)
        self.proposal_chain, self.proposal_accepted = ray.get(proposal_process)
        
        if isinstance(self.proposal, MultipleTry):
            self.proposal.close_pool()
        
    def sample(self, iterations):
        
        if isinstance(self.proposal, MultipleTry):
            self.proposal.open_pool()
            
        # begin iteration
        pbar = tqdm(range(iterations))
        for i in pbar:
            pbar.set_description('Running chain, \u03B1_c = {0:.3f}, \u03B1_f = {1:.2f}'.format(np.mean(self.accepted_coarse[-int(100*self.subsampling_rate):]), np.mean(self.accepted_fine[-100:])))
            
            self.chain_coarse.extend(self.proposal_chain)
            self.accepted_coarse.extend(self.proposal_accepted)
            self.is_coarse.extend([False] + self.subsampling_rate*[True])
            
            for i in range(-self.subsampling_rate+1, 0): 
                self.proposal.adapt(parameters=self.chain_coarse[i].parameters, accepted=list(compress(self.accepted_coarse[:i], self.is_coarse[:i])))
            
            fine_process = fine_worker.remote(self.chain_coarse[-1].parameters, self.link_factory_fine)
            optimistic_process = coarse_worker.remote(self.chain_coarse[-1], self.link_factory_coarse, self.proposal, self.subsampling_rate, True)
            pessimistic_process = coarse_worker.remote(self.chain_coarse[(-self.subsampling_rate+1)], self.link_factory_coarse, self.proposal, self.subsampling_rate, False)
            
            proposal_link_fine = ray.get(fine_process)
            
            alpha = np.exp(proposal_link_fine.posterior - self.chain_fine[-1].posterior + self.chain_coarse[(-self.subsampling_rate+1)].posterior - self.chain_coarse[-1].posterior)   
            
            if np.random.random() < alpha:
                self.chain_fine.append(proposal_link_fine)
                self.accepted_fine.append(True)
                coarse_result = ray.get(optimistic_process)
                self.proposal_chain = coarse_result[0]
                self.proposal_accepted = coarse_result[1]
            else:
                self.chain_fine.append(self.chain_fine[-1])
                self.accepted_fine.append(False)
                coarse_result = ray.get(pessimistic_process)
                self.proposal_chain = coarse_result[0]
                self.proposal_accepted = coarse_result[1]
            
            # update the adaptive error model.
            if self.adaptive_error_model is not None:
                
                if self.adaptive_error_model == 'state-independent':
                    # for the state-independent AEM, we simply update the 
                    # RecursiveSampleMoments with the difference between
                    # the fine and coarse model output
                    self.model_diff = self.R.dot(self.chain_fine[-1].model_output) - self.proposal_chain[0].model_output
                    self.bias.update(self.model_diff)
                    
                    # and update the likelihood in the coarse link factory.
                    self.link_factory_coarse.likelihood.set_bias(self.bias.get_mu(), self.bias.get_sigma())
            
                elif self.adaptive_error_model == 'state-dependent':
                    # for the state-dependent error model, we want the
                    # difference, corrected with the previous difference
                    # to compute the error covariance.
                    self.model_diff_corrected = self.R.dot(self.chain_fine[-1].model_output) - (self.proposal_chain[0].model_output + self.model_diff)
                    
                    # and the "pure" model difference for the offset.
                    self.model_diff = self.R.dot(self.chain_fine[-1].model_output) - self.proposal_chain[0].model_output
                    
                    # update the ZeroMeanRecursiveSampleMoments with the
                    # corrected difference.
                    self.bias.update(self.model_diff_corrected)
                    
                    # and update the likelihood in the coarse link factory
                    # with the "pure" difference.
                    self.link_factory_coarse.likelihood.set_bias(self.model_diff, self.bias.get_sigma())
                
                for j, link in enumerate(self.proposal_chain):
                    self.proposal_chain[j] = self.link_factory_coarse.update_link(link)
        
        if isinstance(self.proposal, MultipleTry):
            self.proposal.close_pool()


@ray.remote
def coarse_worker(initial_link, link_factory, proposal, subsampling_rate, initial_accepted):
    
    chain = [initial_link]
    accepted = [initial_accepted]
    
    for i in range(subsampling_rate):

        # draw a new proposal, given the previous parameters.
        proposal_parameters = proposal.make_proposal(chain[-1])
        
        # create a link from that proposal.
        proposal_link = link_factory.create_link(proposal_parameters)
        
        # compute the acceptance probability, which is unique to
        # the proposal.
        alpha = proposal.get_acceptance(proposal_link, chain[-1])
        
        # perform Metropolis adjustment.
        if np.random.random() < alpha:
            chain.append(proposal_link)
            accepted.append(True)
        else:
            chain.append(chain[-1])
            accepted.append(False)
            
    return chain, accepted

@ray.remote    
def fine_worker(parameters, link_factory):
    return link_factory.create_link(parameters)
