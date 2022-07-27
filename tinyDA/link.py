import numpy as np
from numpy.linalg import det, inv


class Link:

    """The Link class holds all relevant information about an MCMC sample, i.e.
    parameters, prior log-desnity, model output, log-likelihood and possibly a
    Quantity of Interest (QoI)

    Attributes
    ----------
    parameters : numpy.ndarray
        The parameters used to generate the sample
    prior : float
        The prior log-density
    model_output : numpy.ndarray
        The model output
    likelihood : float
        The log-likelihood of the data, given the parameters.
    qoi
        A Quantity of Interest.
    posterior : float
        The (unnormalised) posterior density.
    """

    def __init__(self, parameters, prior, model_output, likelihood, qoi=None):

        """
        Parameters
        ----------
        parameters : numpy.ndarray
            The parameters used to generate the sample
        prior : float
            The prior log-density
        model_output : numpy.ndarray
            The model output
        likelihood : float
            The log-likelihood of the data, given the parameters.
        qoi : optional
            A Quantity of Interest. Default is None
        """

        # internalise parameters.
        self.parameters = parameters
        self.prior = prior
        self.model_output = model_output
        self.likelihood = likelihood
        self.qoi = qoi

        # compute the (unnormalised) posterior.
        self.posterior = self.prior + self.likelihood


class LinkFactory:

    """LinkFactory connects the prior, likelihood and model to produce MCMC
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
        the only requirement is that it has a logpdf method.

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
    evaluate_model(parameters)
        This is the key method of the class. It must be overwritten through
        inheritence to sample a given problem. Given some input parameters,
        it must return a tuple (model_ouput, qoi). The qoi can be None.
    """

    def __init__(self, prior, likelihood):
        """
        Parameters
        ----------
        prior : scipy.stats.rv_continuous
            The prior distribution.
        likelihood : scipy.stats.rv_continuous or tinyDA.LogLike
            The likelihood function.
        """
        # internatlise the distributions.
        self.prior = prior
        self.likelihood = likelihood

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
        model_output, qoi = self.evaluate_model(parameters)

        # compute the likelihood.
        likelihood = self.likelihood.logpdf(model_output)

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
            likelihood = self.likelihood.logpdf(link.model_output)
        else:
            likelihood = self.likelihood.logpdf_custom_bias(link.model_output, bias)

        return Link(
            link.parameters, link.prior, link.model_output, likelihood, link.qoi
        )

    def evaluate_model(self, parameters):
        """
        Parameters
        ----------
        parameters : numpy.ndarray
            A numpy array of model parameters.

        Returns
        ----------
        tuple
            A tuple (model_output, qoi), where model_output is a numpy.ndarray,
            and qoi can be anything, including None.
        """
        # model output must return model_output and qoi (can be None),
        # and must be adapted to the problem at hand.
        model_output = None
        qoi = None
        return model_output, qoi


class BlackBoxLinkFactory(LinkFactory):
    """BlackBoxLinkFactory is a type of LinkFactory, specifically intended for
    use with a black box model. It works the same way as the LinkFactory, but
    addtionally takes 'model' as input, allowing use with generic black box
    models. The model must be either a function that returns the model output
    F(theta), or an object with an equivalent __call__-method. If a Quantity of
    Interest is required, the model must be an object with an additional
    get_qoi-method.

    Attributes
    ----------
    model : Object or function
        A model object or function that returns model output. If model is
        an object, it must have a __call__ method.
    prior : scipy.stats.rv_continuous
        The prior distribution. Usually a scipy.stats.rv_continuous, but
        the only requirement is that it has a logpdf method.
    likelihood : scipy.stats.rv_continuous or tinyDA.LogLike
        The likelihood function. Usually a tinyDA.LogLike, but
        the only requirement is that it has a logpdf method.
    get_qoi : bool
        Whether the model should return a QoI (requires get_qoi method),
        or not. If not, the QoI is None.

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
    evaluate_model(parameters)
        Returns the model outout (and possibly a QoI), given input parameters.
    """

    def __init__(self, model, prior, likelihood, get_qoi=False):
        """
        Parameters
        ----------
        model : Object or function
            A model object or function that returns model output. If model is
            an object, it must have a __call__ method.
        prior : scipy.stats.rv_continuous
            The prior distribution.
        likelihood : scipy.stats.rv_continuous or tinyDA.LogLike
            The likelihood function.
        get_qoi : bool, optional
            Whether the model should return a QoI (requires get_qoi method),
            or not. If not, the QoI is None. Default is False.
        """

        # Internatlise the model
        self.model = model

        # internatlise the distributions.
        self.prior = prior
        self.likelihood = likelihood

        self.get_qoi = get_qoi

    def evaluate_model(self, parameters):
        """
        Parameters
        ----------
        parameters : numpy.ndarray
            A numpy array of model parameters.

        Returns
        ----------
        tuple
            A tuple (model_output, qoi), where model_output is a numpy.ndarray,
            and qoi can be anything.
        """

        # get the model output.
        model_output = self.model(parameters)

        # get the quantity of interest.
        if self.get_qoi:
            qoi = self.model.get_qoi()
        else:
            qoi = None

        # return everything.
        return model_output, qoi
