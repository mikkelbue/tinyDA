import numpy as np
from numpy.linalg import det, inv

class Link:
    
    '''
    The Link class contains all the attributes used by the MCMC.
    '''
    
    def __init__(self, parameters, prior, model_output, likelihood, qoi=None):
        
        # internalise parameters.
        self.parameters = parameters
        self.prior = prior
        self.model_output = model_output
        self.likelihood = likelihood
        self.qoi = qoi
        
        # compute the (unnormalised) posterior.
        self.posterior = self.prior + self.likelihood
        
class DummyLink(Link):
    def __init__(self, parameters):
        self.parameters = parameters

class LinkFactory:
    
    '''
    LinkFactory produces Links. The create_link method calls evaluate_model,
    which is really the key method in this class. It must be overwritten
    through inheritance to sample a problem.
    '''
    
    def __init__(self, prior, likelihood):
        # internatlise the distributions.
        self.prior = prior
        self.likelihood = likelihood
        
    def create_link(self, parameters):
        
        # compute the prior of the parameters.
        prior = self.prior.logpdf(parameters)
        
        # get the model output and the qoi.
        model_output, qoi = self.evaluate_model(parameters)
        
        # compute the likelihood.
        likelihood = self.likelihood.logpdf(model_output)
        
        return Link(parameters, prior, model_output, likelihood, qoi)
        
    def update_link(self, link, bias=None):
        
        if bias is None:
            # recompute the likelihood.
            likelihood = self.likelihood.logpdf(link.model_output)
        else:
            likelihood = self.likelihood.logpdf_custom_bias(link.model_output, bias)
        
        return Link(link.parameters, link.prior, link.model_output, likelihood, link.qoi)
        
    def evaluate_model(self, parameters):
        # model output must return model_output and qoi (can be None),
        # and must be adapted to the problem at hand.
        model_output = None
        qoi = None
        return model_output, qoi
        
class BlackBoxLinkFactory(LinkFactory):
    def __init__(self, model, datapoints, prior, likelihood, get_qoi=False):
        
        # Internatlise the model and datapoints
        self.model = model
        self.datapoints = datapoints
        
        # internatlise the distributions.
        self.prior = prior
        self.likelihood = likelihood
        
        self.get_qoi = get_qoi
    
    def evaluate_model(self, parameters):
        
        # solve the model using the parameters.
        self.model.solve(parameters)
        
        # get the model output at the datapoints.
        output = self.model.get_data(self.datapoints)
        
        # get the quantity of interest.
        if self.get_qoi:
            qoi = self.model.get_qoi()
        else:
            qoi = None
            
        # return everything.
        return output, qoi
