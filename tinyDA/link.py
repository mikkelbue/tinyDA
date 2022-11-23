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
