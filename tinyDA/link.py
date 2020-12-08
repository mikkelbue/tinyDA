import numpy as np
from numpy.linalg import det, inv

class Link:
    
    '''
    The Link class contains all the attributes used by the MCMC.
    There are also methods to compute priors, likelihoods and the (unnormalised) posterior.
    '''
    
    def __init__(self, parameters, prior, model_output, likelihood, qoi=None):
        
        # Set parameters.
        self.parameters = parameters
        self.prior = prior
        self.model_output = model_output
        self.likelihood = likelihood
        qoi = qoi
        
        self.posterior = self.prior + self.likelihood

class LinkFactory:
    def __init__(self, prior, likelihood):
        self.prior = prior
        self.likelihood = likelihood
        
    def create_link(self, parameters):
        prior = self.prior.pdf(parameters)
        model_output, qoi = self.evaluate_model(parameters)
        likelihood = self.likelihood.pdf(model_output)
        return Link(parameters, prior, model_output, likelihood, qoi)
        
    def evaluate_model(self, parameters):
        model_output = None
        qoi = None
        return model_output, qoi
        
