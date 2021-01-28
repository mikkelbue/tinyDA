# externa; imports
import numpy as np
from tqdm import tqdm

# internal imports
from .proposal import *
from .utils import *

class Chain:
    
    '''
    Chain is a single level MCMC sampler. It is initialsed with a
    LinkFactory (which holds the model and the distributions, and returns
    Links), and a proposal (transition kernel).
    '''
    
    def __init__(self, link_factory, proposal, initial_parameters=None):
        
        # internalise the link factory and the proposal
        self.link_factory = link_factory
        self.proposal = proposal
        
        # if the proposal is pCN, Check if the proposal covariance is equal 
        # to the prior covariance and if the prior is zero mean.
        if isinstance(self.proposal, CrankNicolson):
            if not np.allclose(self.link_factory.prior.cov, self.proposal.C):
                raise ValueError('C-proposal must equal C-prior for pCN proposal')
            if np.count_nonzero(self.link_factory.prior.mean):
                raise ValueError('Prior must be zero mean for pCN proposal')
        
        # initialise a list, which holds the links.
        self.chain = []
        
        # initialise a list, which holds boolean acceptance values.
        self.accepted = []
        
        # if the initial parameters are given, use them. otherwise,
        # draw a random sample from the prior.
        if initial_parameters is not None:
            self.initial_parameters = initial_parameters
        else:
            self.initial_parameters = self.link_factory.prior.rvs()
        
        # append a link with the initial parameters (this will automatically
        # compute the model output, and the relevant probabilties.
        self.chain.append(self.link_factory.create_link(self.initial_parameters))
        self.accepted.append(True)
        
        # if the proposal is AM, use the initial parameters as the first
        # sample for the RecursiveSamplingMoments.
        if isinstance(self.proposal, AdaptiveMetropolis):
            self.proposal.initialise_sampling_moments(self.initial_parameters)
        
    def sample(self, iterations):
        
        # start the iteration
        pbar = tqdm(range(iterations))
        for i in pbar:
            pbar.set_description('Running chain, \u03B1 = %0.3f' % np.mean(self.accepted[-100:]))
            
            # draw a new proposal, given the previous parameters.
            proposal = self.proposal.make_proposal(self.chain[-1])
            
            # create a link from that proposal.
            proposal_link = self.link_factory.create_link(proposal)
            
            # compute the acceptance probability, which is unique to
            # the proposal.
            alpha = self.proposal.get_acceptance_ratio(proposal_link, self.chain[-1])
            
            # perform Metropolis adjustment.
            if np.random.random() < alpha:
                self.chain.append(proposal_link)
                self.accepted.append(True)
            else:
                self.chain.append(self.chain[-1])
                self.accepted.append(False)
            
            # adapt the proposal. if the proposal is set to non-adaptive,
            # this has no effect.
            self.proposal.adapt(parameters=self.chain[-1].parameters, accepted=self.accepted)

class DAChain:
    
    '''
    DAChain is a two-level Delayed Acceptance sampler with finite length
    subchains. It takes a coarse and a fine link factory as input, as well
    as a proposal, which applies to the coarse level only.
    The adaptive_error_model can be set to either 'state-independent' or
    'state-dependent'. If the adaptive_error_model is used, the likelihood
    must have a set_bias() method. See example in the distributions.py.
    '''
    
    def __init__(self, link_factory_coarse, link_factory_fine, proposal, initial_parameters=None, adaptive_error_model=None):
        
        # internalise link factories and the proposal
        self.link_factory_coarse = link_factory_coarse
        self.link_factory_fine = link_factory_fine
        self.proposal = proposal
        
        # if the proposal is pCN, Check if the proposal covariance is equal 
        # to the prior covariance and if the prior is zero mean.
        if isinstance(self.proposal, CrankNicolson):
            if not np.allclose(self.link_factory_coarse.prior.cov, self.proposal.C):
                raise ValueError('C-proposal must equal C-prior for pCN proposal')
            if np.count_nonzero(self.link_factory_coarse.prior.mean):
                raise ValueError('Prior must be zero mean for pCN proposal')
                
        # set up lists to hold coarse and fine links, as well as acceptance
        # accounting
        self.chain_coarse = []
        self.accepted_coarse = []
        self.chain_fine = []
        self.accepted_fine = []
                
        # if the initial parameters are given, use them. otherwise,
        # draw a random sample from the prior.
        if initial_parameters is not None:
            self.initial_parameters = initial_parameters
        else:
            self.initial_parameters = self.link_factory_coarse.prior.rvs()
            
        # append a link with the initial parameters to the coarse chain.
        self.chain_coarse.append(self.link_factory_coarse.create_link(self.initial_parameters))
        self.accepted_coarse.append(True)
        
        # append a link with the initial parameters to the coarse chain.
        self.chain_fine.append(self.link_factory_fine.create_link(self.initial_parameters))
        self.accepted_fine.append(True)
        
        # if the proposal is AM, use the initial parameters as the first
        # sample for the RecursiveSamplingMoments.
        if isinstance(self.proposal, AdaptiveMetropolis):
            self.proposal.initialise_sampling_moments(self.initial_parameters)
        
        # set the adative error model as a. attribute.
        self.adaptive_error_model = adaptive_error_model
        
        # set up the adaptive error model.
        if self.adaptive_error_model is not None:
            
            # compute the difference between coarse and fine level.
            self.model_diff = self.chain_fine[-1].model_output - self.chain_coarse[-1].model_output
            
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
        
    def sample(self, iterations, subsampling_rate):
            
        # begin iteration
        pbar = tqdm(range(iterations))
        for i in pbar:
            pbar.set_description('Running chain, \u03B1 = %0.3f' % np.mean(self.accepted_fine[-100:]))
            
            # subsample the coarse model.
            for j in range(subsampling_rate):
                
                # draw a new proposal, given the previous parameters.
                proposal = self.proposal.make_proposal(self.chain_coarse[-1])
                
                # create a link from that proposal.
                proposal_link_coarse = self.link_factory_coarse.create_link(proposal)
                
                # compute the acceptance probability, which is unique to
                # the proposal.
                alpha_1 = self.proposal.get_acceptance_ratio(proposal_link_coarse, self.chain_coarse[-1])
                
                # perform Metropolis adjustment.
                if np.random.random() < alpha_1:
                    self.chain_coarse.append(proposal_link_coarse)
                    self.accepted_coarse.append(True)
                else:
                    self.chain_coarse.append(self.chain_coarse[-1])
                    self.accepted_coarse.append(False)
                
                # adapt the proposal. if the proposal is set to non-adaptive,
                # this has no effect.
                self.proposal.adapt(parameters=self.chain_coarse[-1].parameters, accepted=self.accepted_coarse)
            
            # when subsampling is complete, create a new fine link from the
            # previous coarse link.
            proposal_link_fine = self.link_factory_fine.create_link(self.chain_coarse[-1].parameters)
            
            # compute the delayed acceptance probability.
            alpha_2 = np.exp(proposal_link_fine.likelihood - self.chain_fine[-1].likelihood + self.chain_coarse[-subsampling_rate].likelihood - self.chain_coarse[-1].likelihood)
            
            # Perform Metropolis adjustment, and update the coarse chain
            # to restart from the previous accepted fine link.
            if np.random.random() < alpha_2:
                self.chain_fine.append(proposal_link_fine)
                self.accepted_fine.append(True)
                self.chain_coarse.append(self.chain_coarse[-1])
                self.accepted_coarse.append(True)
            else:
                self.chain_fine.append(self.chain_fine[-1])
                self.accepted_fine.append(False)
                self.chain_coarse.append(self.chain_coarse[-(subsampling_rate+1)])
                self.accepted_coarse.append(False)
            
            # update the adaptive error model.
            if self.adaptive_error_model is not None:
                
                if self.adaptive_error_model == 'state-independent':
                    # for the state-independent AEM, we simply update the 
                    # RecursiveSampleMoments with the difference between
                    # the fine and coarse model output
                    self.model_diff = self.chain_fine[-1].model_output - self.chain_coarse[-1].model_output
                    self.bias.update(self.model_diff)
                    
                    # and update the likelihood in the coarse link factory.
                    self.link_factory_coarse.likelihood.set_bias(self.bias.get_mu(), self.bias.get_sigma())
            
                elif self.adaptive_error_model == 'state-dependent':
                    # for the state-dependent error model, we want the
                    # difference, corrected with the previous difference
                    # to compute the error covariance.
                    self.model_diff_corrected = self.chain_fine[-1].model_output - (self.chain_coarse[-1].model_output + self.model_diff)
                    
                    # and the "pure" model difference for the offset.
                    self.model_diff = self.chain_fine[-1].model_output - self.chain_coarse[-1].model_output
                    
                    # update the ZeroMeanRecursiveSampleMoments with the
                    # corrected difference.
                    self.bias.update(self.model_diff_corrected)
                    
                    # and update the likelihood in the coarse link factory
                    # with the "pure" difference.
                    self.link_factory_coarse.likelihood.set_bias(self.model_diff, self.bias.get_sigma())
