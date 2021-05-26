import numpy as np
from numpy.linalg import det, inv

class Link:
    
    '''
    The Link class contains all the attributes used by the MCMC.
    '''
    
    def __init__(self, parameters, prior, model_output, likelihood, gradient=None, qoi=None):
        
        # internalise parameters.
        self.parameters = parameters
        self.prior = prior
        self.model_output = model_output
        self.likelihood = likelihood
        self.gradient = gradient
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
    
    compute_gradient = False
    
    def __init__(self, prior, likelihood):
        # internatlise the distributions.
        self.prior = prior
        self.likelihood = likelihood
        
    def create_link(self, parameters, child=False):
        
        # compute the prior of the parameters.
        prior = self.prior.logpdf(parameters)
        
        # get the model output and the qoi.
        model_output, qoi = self.evaluate_model(parameters)
        
        # compute the likelihood.
        likelihood = self.likelihood.logpdf(model_output)
        
        if self.compute_gradient and not child:
            gradient = self.evaluate_gradient(parameters, prior+likelihood)
        else:
            gradient = None
        
        return Link(parameters, prior, model_output, likelihood, gradient, qoi)
        
    def update_link(self, link, bias=None):
        
        if bias is None:
            # recompute the likelihood.
            likelihood = self.likelihood.logpdf(link.model_output)
        else:
            likelihood = self.likelihood.logpdf_custom_bias(link.model_output, bias)
            
        if self.compute_gradient:
            gradient = self.evaluate_gradient(link.parameters, link.prior+likelihood)
        else:
            gradient = None
        
        return Link(link.parameters, link.prior, link.model_output, likelihood, gradient, link.qoi)
        
    def evaluate_model(self, parameters):
        # model output must return model_output and qoi (can be None),
        # and must be adapted to the problem at hand.
        model_output = None
        qoi = None
        return model_output, qoi
        
    def evaluate_gradient(self, parameters, posterior):
        dim = parameters.shape[0]
        gradient = np.zeros(dim)
        for i in range(dim):
            pertubation = np.zeros(dim)
            pertubation[i] = 1e-3
            gradient[i] = (self.create_link(parameters+pertubation, True).posterior - posterior)/1e-3
        return gradient
        
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
