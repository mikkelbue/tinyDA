import numpy as np

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
    qoi : any
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
        prior : float or 1-element array-like
            The prior log-density
        model_output : numpy.ndarray
            The model output
        likelihood : float or 1-element array-like
            The log-likelihood of the data, given the parameters.
        qoi : optional
            A Quantity of Interest. Default is None
        """

        # internalise parameters.
        self.parameters = parameters
        self.prior = _to_scalar(prior, "prior")
        self.model_output = model_output
        self.likelihood = _to_scalar(likelihood, "likelihood")
        self.qoi = qoi

        # compute the (unnormalised) posterior.
        self.posterior = self.prior + self.likelihood

def _to_scalar(x, name):
    arr = np.asarray(x)
    if arr.ndim > 0 and arr.size != 1:
        raise ValueError(
            f"{name} must be a scalar or length-1 array, got shape {arr.shape}"
        )
    return arr.item()

