# externa; imports
from itertools import compress
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
        
        # if the proposal is pCN, Check if the proposal covariance is equal 
        # to the prior covariance and if the prior is zero mean.
        if isinstance(proposal, CrankNicolson) and not isinstance(link_factory.prior, stats._multivariate.multivariate_normal_frozen):
            raise TypeError('Prior must be of type scipy.stats.multivariate_normal for pCN proposal')
        
        # check the same if the CrankNicolson is nested in a MultipleTry or GaussianTransportMap proposal.
        elif isinstance(proposal, MultipleTry) or isinstance(proposal, GaussianTransportMap):
            if isinstance(proposal.kernel, CrankNicolson) and not isinstance(link_factory.prior, stats._multivariate.multivariate_normal_frozen):
                raise TypeError('Prior must be of type scipy.stats.multivariate_normal for pCN kernel')
                
        if isinstance(proposal, MALA):
            link_factory.compute_gradient = True
        
        # internalise the link factory and the proposal
        self.link_factory = link_factory
        self.proposal = proposal
        
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
        
        # setup the proposal
        self.proposal.setup_proposal(parameters=self.initial_parameters, link_factory=self.link_factory)
        
    def sample(self, iterations):
        
        if isinstance(self.proposal, MultipleTry):
            self.proposal.open_pool()
        
        # start the iteration
        pbar = tqdm(range(iterations))
        for i in pbar:
            pbar.set_description('Running chain, \u03B1 = %0.2f' % np.mean(self.accepted[-100:]))
            
            # draw a new proposal, given the previous parameters.
            proposal = self.proposal.make_proposal(self.chain[-1])
            
            # create a link from that proposal.
            proposal_link = self.link_factory.create_link(proposal)
            
            # compute the acceptance probability, which is unique to
            # the proposal.
            alpha = self.proposal.get_acceptance(proposal_link, self.chain[-1])
            
            # perform Metropolis adjustment.
            if np.random.random() < alpha:
                self.chain.append(proposal_link)
                self.accepted.append(True)
            else:
                self.chain.append(self.chain[-1])
                self.accepted.append(False)
            
            # adapt the proposal. if the proposal is set to non-adaptive,
            # this has no effect.
            self.proposal.adapt(parameters=self.chain[-1].parameters, 
                                jumping_distance=self.chain[-1].parameters-self.chain[-2].parameters, 
                                accepted=self.accepted)
        
        if isinstance(self.proposal, MultipleTry):
            self.proposal.close_pool()

class DAChain:
    
    '''
    DAChain is a two-level Delayed Acceptance sampler with finite length
    subchains. It takes a coarse and a fine link factory as input, as well
    as a proposal, which applies to the coarse level only.
    The adaptive_error_model can be set to either 'state-independent' or
    'state-dependent'. If the adaptive_error_model is used, the likelihood
    must have a set_bias() method. See example in the distributions.py.
    '''
    
    def __init__(self, link_factory_coarse, link_factory_fine, proposal, subsampling_rate=1, initial_parameters=None, adaptive_error_model=None, R=None):
        
        # if the proposal is pCN, Check if the proposal covariance is equal 
        # to the prior covariance and if the prior is zero mean.
        if isinstance(proposal, CrankNicolson) and not isinstance(link_factory_coarse.prior, stats._multivariate.multivariate_normal_frozen):
            raise TypeError('Prior must be of type scipy.stats.multivariate_normal for pCN proposal')
        
        # check the same if the CrankNicolson is nested in a MultipleTry or GaussianTransportMap proposal.
        elif isinstance(proposal, MultipleTry) or isinstance(proposal, GaussianTransportMap):
            if isinstance(proposal.kernel, CrankNicolson) and not isinstance(link_factory_coarse.prior, stats._multivariate.multivariate_normal_frozen):
                raise TypeError('Prior must be of type scipy.stats.multivariate_normal for pCN kernel')
                
        if isinstance(proposal, MALA):
            if adaptive_error_model == 'state-dependent':
                raise NotImplementedError('MALA proposal is not compatible with state-dependent error model')
            link_factory_coarse.compute_gradient = True
        
        # internalise link factories and the proposal
        self.link_factory_coarse = link_factory_coarse
        self.link_factory_fine = link_factory_fine
        self.proposal = proposal
        self.subsampling_rate = subsampling_rate
                
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
            
        # append a link with the initial parameters to the coarse chain.
        self.chain_coarse.append(self.link_factory_coarse.create_link(self.initial_parameters))
        self.accepted_coarse.append(True)
        self.is_coarse.append(False)
        
        # append a link with the initial parameters to the coarse chain.
        self.chain_fine.append(self.link_factory_fine.create_link(self.initial_parameters))
        self.accepted_fine.append(True)
        
        # setup the proposal
        self.proposal.setup_proposal(parameters=self.initial_parameters, link_factory=self.link_factory_coarse)
        
        # set the adative error model as a. attribute.
        self.adaptive_error_model = adaptive_error_model
        
        # set up the adaptive error model.
        if self.adaptive_error_model is not None:
            
            if R is None:
                self.R = np.eye(self.chain_fine[-1].model_output.shape[0])
            else:
                self.R = R
            
            # compute the difference between coarse and fine level.
            self.model_diff = self.R.dot(self.chain_fine[-1].model_output) - self.chain_coarse[-1].model_output
            
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
        
            self.chain_coarse[-1] = self.link_factory_coarse.update_link(self.chain_coarse[-1])
        
    def sample(self, iterations):
        
        if isinstance(self.proposal, MultipleTry):
            self.proposal.open_pool()
            
        # begin iteration
        pbar = tqdm(range(iterations))
        for i in pbar:
            pbar.set_description('Running chain, \u03B1_c = {0:.3f}, \u03B1_f = {1:.2f}'.format(np.mean(self.accepted_coarse[-int(100*self.subsampling_rate):]), np.mean(self.accepted_fine[-100:])))
            
            # subsample the coarse model.
            for j in range(self.subsampling_rate):
                
                # draw a new proposal, given the previous parameters.
                proposal = self.proposal.make_proposal(self.chain_coarse[-1])
                
                # create a link from that proposal.
                proposal_link_coarse = self.link_factory_coarse.create_link(proposal)
                
                # compute the acceptance probability, which is unique to
                # the proposal.
                alpha_1 = self.proposal.get_acceptance(proposal_link_coarse, self.chain_coarse[-1])
                
                # perform Metropolis adjustment.
                if np.random.random() < alpha_1:
                    self.chain_coarse.append(proposal_link_coarse)
                    self.accepted_coarse.append(True)
                    self.is_coarse.append(True)
                else:
                    self.chain_coarse.append(self.chain_coarse[-1])
                    self.accepted_coarse.append(False)
                    self.is_coarse.append(True)
                
                # adapt the proposal. if the proposal is set to non-adaptive,
                # this has no effect.
                self.proposal.adapt(parameters=self.chain_coarse[-1].parameters, 
                                    jumping_distance=self.chain_coarse[-1].parameters-self.chain_coarse[-2].parameters, 
                                    accepted=list(compress(self.accepted_coarse, self.is_coarse)))
            
            if sum(self.accepted_coarse[-self.subsampling_rate:]) == 0:
                self.chain_fine.append(self.chain_fine[-1])
                self.accepted_fine.append(False)
                self.chain_coarse.append(self.chain_coarse[-(self.subsampling_rate+1)])
                self.accepted_coarse.append(False)
                self.is_coarse.append(False)
                
            else:
                # when subsampling is complete, create a new fine link from the
                # previous coarse link.
                proposal_link_fine = self.link_factory_fine.create_link(self.chain_coarse[-1].parameters)
                
                # compute the delayed acceptance probability.
                if self.adaptive_error_model == 'state-dependent':
                    
                    bias_next = self.R.dot(proposal_link_fine.model_output) - self.chain_coarse[-1].model_output
                    coarse_state_biased = self.link_factory_coarse.update_link(self.chain_coarse[-(self.subsampling_rate+1)], bias_next)
                    
                    if self.proposal.is_symmetric:
                        q_x_y = q_y_x = 0
                    else:
                        q_x_y = self.proposal.get_q(self.chain_fine[-1], proposal_link_fine)
                        q_y_x = self.proposal.get_q(proposal_link_fine, self.chain_fine[-1])
                    
                    alpha_2 = np.exp(min(proposal_link_fine.posterior + q_y_x, coarse_state_biased.posterior + q_x_y) - min(self.chain_fine[-1].posterior + q_x_y, self.chain_coarse[-1].posterior + q_y_x))
                
                else:
                    alpha_2 = np.exp(proposal_link_fine.posterior - self.chain_fine[-1].posterior + self.chain_coarse[-(self.subsampling_rate+1)].posterior - self.chain_coarse[-1].posterior)
                
                # Perform Metropolis adjustment, and update the coarse chain
                # to restart from the previous accepted fine link.
                if np.random.random() < alpha_2:
                    self.chain_fine.append(proposal_link_fine)
                    self.accepted_fine.append(True)
                    self.chain_coarse.append(self.chain_coarse[-1])
                    self.accepted_coarse.append(True)
                    self.is_coarse.append(False)
                else:
                    self.chain_fine.append(self.chain_fine[-1])
                    self.accepted_fine.append(False)
                    self.chain_coarse.append(self.chain_coarse[-(self.subsampling_rate+1)])
                    self.accepted_coarse.append(False)
                    self.is_coarse.append(False)
            
            # update the adaptive error model.
            if self.adaptive_error_model is not None:
                
                if self.adaptive_error_model == 'state-independent':
                    # for the state-independent AEM, we simply update the 
                    # RecursiveSampleMoments with the difference between
                    # the fine and coarse model output
                    self.model_diff = self.R.dot(self.chain_fine[-1].model_output) - self.chain_coarse[-1].model_output
                    self.bias.update(self.model_diff)
                    
                    # and update the likelihood in the coarse link factory.
                    self.link_factory_coarse.likelihood.set_bias(self.bias.get_mu(), self.bias.get_sigma())
            
                elif self.adaptive_error_model == 'state-dependent':
                    # for the state-dependent error model, we want the
                    # difference, corrected with the previous difference
                    # to compute the error covariance.
                    self.model_diff_corrected = self.R.dot(self.chain_fine[-1].model_output) - (self.chain_coarse[-1].model_output + self.model_diff)
                    
                    # and the "pure" model difference for the offset.
                    self.model_diff = self.R.dot(self.chain_fine[-1].model_output) - self.chain_coarse[-1].model_output
                    
                    # update the ZeroMeanRecursiveSampleMoments with the
                    # corrected difference.
                    self.bias.update(self.model_diff_corrected)
                    
                    # and update the likelihood in the coarse link factory
                    # with the "pure" difference.
                    self.link_factory_coarse.likelihood.set_bias(self.model_diff, self.bias.get_sigma())
                
                self.chain_coarse[-1] = self.link_factory_coarse.update_link(self.chain_coarse[-1])
        
        if isinstance(self.proposal, MultipleTry):
            self.proposal.close_pool()
            
