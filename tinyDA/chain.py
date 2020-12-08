import numpy as np
from tqdm import tqdm

#from .link import *
#from .distributions import *
from .proposal import *
#from .diagnostics import *
from .utils import *

class Chain:
    def __init__(self, link_factory, proposal):
        
        self.link_factory = link_factory
        self.proposal = proposal
        
        self.chain = []
        self.accepted = []
        
    def sample(self, iterations, initial_parameters=None):
        
        if initial_parameters is not None:
            self.initial_parameters = initial_parameters
        else:
            self.initial_parameters = self.link_factory.prior.rvs()
            
        if isinstance(self.proposal, AdaptiveMetropolis):
            self.proposal.initialise_sampling_moments(self.initial_parameters)
            
        self.chain.append(self.link_factory.create_link(self.initial_parameters))
        self.accepted.append(True)
        
        pbar = tqdm(range(iterations))
        for i in pbar:
            pbar.set_description('Running chain, \u03B1 = %0.3f' % np.mean(self.accepted[-100:]))
            
            proposal = self.proposal.make_proposal(self.chain[-1].parameters)
            proposal_link = self.link_factory.create_link(proposal)
            alpha = self.proposal.get_acceptance_ratio(proposal_link, self.chain[-1])
            
            if np.random.random() < alpha:
                self.chain.append(proposal_link)
                self.accepted.append(True)
            else:
                self.chain.append(self.chain[-1])
                self.accepted.append(False)
                
            self.proposal.adapt(parameters=self.chain[-1].parameters, accepted=self.accepted)

class DAChain:
    def __init__(self, link_factory_coarse, link_factory_fine, proposal, adaptive_error_model=None):
        
        self.link_factory_coarse = link_factory_coarse
        self.link_factory_fine = link_factory_fine
        self.proposal = proposal
        
        self.chain_coarse = []
        self.accepted_coarse = []
        self.chain_fine = []
        self.accepted_fine = []
        
        self.adaptive_error_model = adaptive_error_model
        
    def sample(self, iterations, subsampling_rate, initial_parameters=None):
        
        if initial_parameters is not None:
            self.initial_parameters = initial_parameters
        else:
            self.initial_parameters = self.link_factory_coarse.prior.rvs()
            
        if isinstance(self.proposal, AdaptiveMetropolis):
            self.proposal.initialise_sampling_moments(self.initial_parameters)
            
        self.chain_coarse.append(self.link_factory_coarse.create_link(self.initial_parameters))
        self.accepted_coarse.append(True)
        
        self.chain_fine.append(self.link_factory_fine.create_link(self.initial_parameters))
        self.accepted_fine.append(True)
        
        if self.adaptive_error_model is not None:
            
            model_diff = self.chain_fine[-1].model_output - self.chain_coarse[-1].model_output
            
            if self.adaptive_error_model == 'state independent':
                self.bias = RecursiveSampleMoments(model_diff, np.zeros((model_diff.shape[0], model_diff.shape[0])))
                self.link_factory_coarse.likelihood.set_bias(self.bias.get_mu(), self.bias.get_sigma())
            
            elif self.adaptive_error_model == 'state dependent':
                self.bias = ZeroMeanRecursiveSampleMoments(np.zeros((model_diff.shape[0], model_diff.shape[0])))
                self.link_factory_coarse.likelihood.set_bias(model_diff, self.bias.get_sigma())
            
        
        pbar = tqdm(range(iterations))
        for i in pbar:
            pbar.set_description('Running chain, \u03B1 = %0.3f' % np.mean(self.accepted_fine[-100:]))
            
            for j in range(subsampling_rate):
                proposal = self.proposal.make_proposal(self.chain_coarse[-1].parameters)
                proposal_link_coarse = self.link_factory_coarse.create_link(proposal)
                alpha_1 = self.proposal.get_acceptance_ratio(proposal_link_coarse, self.chain_coarse[-1])
            
                if np.random.random() < alpha_1:
                    self.chain_coarse.append(proposal_link_coarse)
                    self.accepted_coarse.append(True)
                else:
                    self.chain_coarse.append(self.chain_coarse[-1])
                    self.accepted_coarse.append(False)
                
                self.proposal.adapt(parameters=self.chain_coarse[-1].parameters, accepted=self.accepted_coarse)
            
            proposal_link_fine = self.link_factory_fine.create_link(self.chain_coarse[-1].parameters)
            alpha_2 = np.exp(proposal_link_fine.likelihood - self.chain_fine[-1].likelihood + self.chain_coarse[-subsampling_rate].likelihood - self.chain_coarse[-1].likelihood)
            
            if np.random.random() < alpha_2:
                self.chain_fine.append(proposal_link_fine)
                self.accepted_fine.append(True)
                self.chain_coarse.append(self.chain_coarse[-1])
            else:
                self.chain_fine.append(self.chain_fine[-1])
                self.accepted_fine.append(False)
                self.chain_coarse.append(self.chain_coarse[-(subsampling_rate+1)])

            if self.adaptive_error_model is not None:
            
                if self.adaptive_error_model == 'state independent':
                    model_diff = self.chain_fine[-1].model_output - self.chain_coarse[-1].model_output
                    
                    self.bias.update(model_diff)
                    self.link_factory_coarse.likelihood.set_bias(self.bias.get_mu(), self.bias.get_sigma())
            
                elif self.adaptive_error_model == 'state dependent':
                    model_diff_corrected = self.chain_fine[-1].model_output - (self.chain_coarse[-1].model_output + model_diff)
                    model_diff = self.chain_fine[-1].model_output - self.chain_coarse[-1].model_output
                    
                    self.bias.update(model_diff_corrected)
                    self.link_factory_coarse.likelihood.set_bias(model_diff, self.bias.get_sigma())
