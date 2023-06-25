import warnings

import numpy as np
from numpy.linalg import inv

from .link import Link

class Posterior:

    """Posterior connects the prior, likelihood and model to produce MCMC
    samples (Links). The create_link-method calls evaluate_model, which is the
    key method in this class. It must be overwritten through inheritance to
    sample a problem and takes a numpy array of model parameters as input and
    outputs a tuple of model output F(theta), Quantity of Interest (qoi). The
    qoi can be None.

    Attributes
    ----------
    prior : scipy.stats.rv_continuous
        The prior distribution. Usually a scipy.stats.rv_continuous, but
        the only requirement is that it has a logpdf method.
    likelihood : scipy.stats.rv_continuous or tinyDA.LogLike
        The likelihood function. Usually a tinyDA.LogLike, but
        the only requirement is that it has a loglike method.
    model : function or callable
        The forward operator, i.e. a map F(theta). This could be a Python
        function or a class instance with a __call__() method.

    Methods
    -------
    create_link(parameters)
        Returns an instance of tinyDA.Link, given the parameters. Calling
        the method with a numpy.ndarray of parameters triggers evaluation
        of the prior density, the model and the likelihood.
    update_link(link, bias=None)
        This is a helper method that updates a link after the likelihood
        function has been adaptively updated. Hence, it skips model evaluation
        and only recomputes the likelihood.
    """

    def __init__(self, prior, likelihood, model=None):
        """
        Parameters
        ----------
        prior : scipy.stats.rv_continuous
            The prior distribution.
        likelihood : scipy.stats.rv_continuous or tinyDA.LogLike
            The likelihood function.
        model : function or callable
            The forward operator, i.e. a map F(theta). This could be a Python
            function or a class instance with a __call__() method.
        """
        # internatlise the distributions.
        self.prior = prior
        self.likelihood = likelihood

        # backwards compatibility.
        if model is None:
            self.model = self.evaluate_model
        else:
            self.model = model

    def create_link(self, parameters):
        """
        Parameters
        ----------
        parameters : numpy.ndarray
            A numpy array of model parameters.

        Returns
        ----------
        tinyDA.Link
            A TinyDA.Link with attributes corresponding to the input parameters.
        """

        # compute the prior of the parameters.
        prior = self.prior.logpdf(parameters)

        # get the model output and the qoi.
        model_output = self.model(parameters)

        # extract model output and qoi.
        if isinstance(model_output, tuple):
            model_output, qoi = model_output
        else:
            qoi = None

        # compute the likelihood.
        likelihood = self.likelihood.loglike(model_output)

        return Link(parameters, prior, model_output, likelihood, qoi)

    def update_link(self, link, bias=None):
        """
        Parameters
        ----------
        link : tinyDA.Link
            A tinyDA.Link that should be updated, if the likelihood
            function has been updated.

        Returns
        ----------
        tinyDA.Link
            A TinyDA.Link with updated likelihood and posterior.
        """

        if bias is None:
            # recompute the likelihood.
            likelihood = self.likelihood.loglike(link.model_output)
        else:
            likelihood = self.likelihood.loglike_custom_bias(link.model_output, bias)

        return Link(
            link.parameters, link.prior, link.model_output, likelihood, link.qoi
        )

class LinkFactory(Posterior):
    def __init__(self, prior, likelihood):

        warnings.warn(
            "LinkFactory is deprecated and will be removed in the next version. Please use Posterior instead",
            stacklevel=2,
        )

        super().__init__(prior, likelihood)

class BlackBoxLinkFactory(Posterior):
    def __init__(self, model, prior, likelihood, get_qoi=False):

        warnings.warn(
            "BlackBoxLinkFactory is deprecated and will be removed in the next version. Please use Posterior instead.",
            stacklevel=2,
        )

        if get_qoi:
            warnings.warn(
                "The argument get_qoi has been removed and has no effect. If a quantity of interest is required, the call the the forward model must return a tuple of (model_output, qoi)",
                stacklevel=2,
            )

        super().__init__(prior, likelihood, model)
