# external imports
import warnings
import numpy as np

import scipy.stats as stats


class JointPrior:

    """JointPrior is a wrapper for a list of priors, if the parameters have
    different types of priors. The order must match the order of parameters for
    the model, since parameters are unnamed.

    Attributes
    ----------
    distributions : list
        A list of distributions, typically each a scipy.stats.rv_continuous.
        Each distribution must have at least a logpdf method.
    dim : int
        The (total) dimensionality of the prior.

    Methods
    ----------
    logpdf(x)
        Returns the sum of log probabilities of the distributions.
    rvs(n_samples=1)
        Returns n_samples random samples from the priors.
    ppf(x)
        Percent point function. Used to transform uniform samples, i.e.
        from a latin hypercube, to the prior distribtions.
    """

    def __init__(self, distributions):
        """
        Parameters
        ----------
        distributions : list
            A list of distributions, typically each a scipy.stats.rv_continuous.
            Each distribution must have at least a logpdf method.
        """
        self.distributions = distributions
        self.dim = len(distributions)

    def logpdf(self, x):
        """
        Parameters
        ----------
        x : numpy.ndarray
            A numpy array of model parameters to be evaluated by the priors.

        Returns
        ----------
        float
            The sum of log-probabilities of the priors.
        """
        return sum([self.distributions[i].logpdf(x[i]) for i in range(self.dim)])

    def rvs(self, n_samples=1):
        """
        Parameters
        ----------
        n_samples : int
            Number of samples drawn from the priors.

        Returns
        ----------
        numpy.ndarray
            An (n_samples x dim) array of random samples from the prior.
        """
        x = np.zeros((n_samples, self.dim))

        for i in range(self.dim):
            x[:, i] = self.distributions[i].rvs(size=n_samples)

        if n_samples == 1:
            return x.flatten()
        else:
            return x

    def ppf(self, x):
        """
        Parameters
        ----------
        x : numpy.ndarray
            A numpy array of model parameters to be transformed using
            the percent point function. Columns correspond to different
            priors.

        Returns
        ----------
        numpy.ndarray
            Transformed parameters.
        """

        y = np.zeros(x.shape)

        for i in range(self.dim):
            y[:, i] = self.distributions[i].ppf(x[:, i])

        return y


def CompositePrior(*args, **kwargs):
    """Deprecation dummy."""
    warnings.warn(" CompositePrior has been deprecated. Please use JointPrior.")
    return JointPrior(*args, **kwargs)


class PoissonPointProcess:

    """PoissonPointProcess is a geometric prior, where the number of points
    has a Poisson distribution and their locations are uniformly distributed
    on the domain. Additional geometric attributes of the points can also be
    assigned.

    Attributes
    ----------
    lamb : float
        The Poisson rate.
    domain : numpy.ndarray
        A d x 2 numpy array describing the bounds of the domain, where
        the first column are the minima of each dimension and the second
        column are the maxima of each dimension.
    attributes : dict
        A dictionary of additional attributes of the points (size, direction,
        etc.), with the format {"attribute": scipy.stats.distribution}
    pois : scipy.stats.poisson
        The Poisson distribution over the number of points.
    domain_dist : scipy.stats.uniform
        A d-dimensional uniform distribution describing the domain.

    Methods
    ----------
    logpdf(x)
        Returns the Poisson pmf of x.
    rvs()
        Generates one sample from the Poisson Point Process prior.
    """

    def __init__(self, lamb, domain, attributes={}):
        """
        Parameters
        ----------
        lamb : float
            The Poisson rate.
        domain : numpy.ndarray
            A d x 2 numpy array describing the bounds of the domain, where
            the first column are the minima of each dimension and the second
            column are the maxima of each dimension.
        attributes : dict
            A dictionary of additional attributes of the points (size, direction,
            etc.), with the format {"attribute": scipy.stats.distribution}
        """

        # set the Poisson rate.
        self.lamb = lamb
        # internalise the domain.
        self.domain = domain
        self.n_dim = domain.shape[0]
        # set the point process attributes.
        self.attributes = attributes

        # initialise a Poisson distribution.
        self.pois = stats.poisson(lamb)

        # initialise a uniform distribution for the domain.
        self.domain_dist = stats.uniform(
            loc=self.domain[:, 0], scale=self.domain[:, 1] - self.domain[:, 0]
        )

    def logpdf(self, x):
        # check the all the points are within the domain distribution.
        positions = np.array([point["position"] for point in x]).T
        for i in range(self.n_dim):
            if not (
                np.all(positions[i, :] >= self.domain[i, 0])
                and np.all(positions[i, :] <= self.domain[i, 1])
            ):
                return -np.inf

        # if the points are within domain, the probability is the Poisson mass
        # over the number of points.
        return self.pois.logpmf(len(x))

    def rvs(self):
        # get a random draw from the Poisson distribution.
        k = self.pois.rvs()

        # create k points.
        return [self._create_point() for i in range(k)]

    def _create_point(self):
        # get a random position for the point.
        point = {"position": self.domain_dist.rvs()}

        # create random attributes for the point.
        for attr, dist in self.attributes.items():
            point[attr] = dist.rvs()

        return point


def GaussianLogLike(data, covariance):
    """Factory pattern for initialising Gaussian likelihoods.
    If a diagonal covariance matrix is provided, this will return a
    DiagonalGaussianLogLike object. If all the diagonal terms are the same,
    it will return an IsotropicGaussianLogLike object. Otherwise it will
    return a DefaultGaussianLogLike.

    Parameters
    ----------
    data : numpy.ndarray
        A 1-D NumPy array of the data.
    covariance : numpy.ndarray
        A 2-D NumPy array of the data covariance matrix.

    Returns
    ----------
    object
        A DefaultGaussianLoglike, DiagonalGaussianLogLike or an
        IsotropicGaussianLogLike, depending on the structure of the
        giver covariance matrix.
    """

    # check if covariance operator is a square numpy array.

    if not isinstance(covariance, np.ndarray):
        raise TypeError("Covariance must be a 2-D numpy array.")
    if covariance.ndim == 2:
        if not covariance.shape[0] == data.shape[0]:
            raise ValueError("Dimensions of data and covariance do not match.")
        if not covariance.shape[0] == covariance.shape[1]:
            raise ValueError("Covariance must be an NxN array.")
    else:
        raise TypeError("Covariance must be a 2-D numpy array.")

    if np.count_nonzero(covariance - np.diag(np.diag(covariance))) == 0:
        if np.all(np.diag(covariance) == covariance[0, 0]):
            return IsotropicGaussianLogLike(data, covariance[0, 0])
        else:
            return DiagonalGaussianLogLike(data, covariance)
    else:
        return DefaultGaussianLogLike(data, covariance)


class DefaultGaussianLogLike:
    """GaussianLogLike is a minimal implementation of the (unnormalised)
    Gaussian likelihood function.

    Attributes
    ----------
    data : numpy.ndarray
        The Gaussian distributed data.
    cov : numpy.ndarray
        The covariance of the Gaussian likelihood function.
    cov_inverse : numpy.ndarray
        The inverse of the covariance.

    Methods
    ----------
    loglike(x)
        Returns the log-likelihood of the input array x.
    """

    def __init__(self, data, covariance):
        """
        Parameters
        ----------
        data : numpy.ndarray
            The Gaussian distributed data.
        cov : numpy.ndarray
            The covariance of the Gaussian likelihood function.
        """

        # set the data and covariance as attributes
        self.data = data
        self.cov = covariance

        # precompute the inverse of the covariance.
        self.cov_inverse = np.linalg.inv(self.cov)

    def loglike(self, x):
        """
        Parameters
        ----------
        x : numpy.ndarray
            Input array, to be evaluated by the likelihood function.

        Returns
        ----------
        float
            The log-likelihood of the input array.
        """

        # compute the unnormalised likelihood.
        return -0.5 * np.linalg.multi_dot(
            ((x - self.data).T, self.cov_inverse, (x - self.data))
        )

    def grad_loglike(self, x):
        return np.dot(self.cov_inverse, (self.data - x))


class DiagonalGaussianLogLike(DefaultGaussianLogLike):
    def __init__(self, data, covariance):
        # set the data and covariance as attributes
        self.data = data
        self.cov = np.diag(covariance)

    def loglike(self, x):
        # compute the unnormalised likelihood.
        return -0.5 * ((x - self.data) ** 2 / self.cov).sum()

    def grad_loglike(self, x):
        return 1 / self.cov * (self.data - x)


class IsotropicGaussianLogLike(DefaultGaussianLogLike):
    def __init__(self, data, variance):
        # set the data and variance as attributes
        self.data = data
        self.var = variance

    def loglike(self, x):
        # compute the unnormalised likelihood.
        return -0.5 * np.linalg.norm(x - self.data) ** 2 / self.var

    def grad_loglike(self, x):
        return 1 / self.var * (self.data - x)


class AdaptiveGaussianLogLike(DefaultGaussianLogLike):
    """AdaptiveLogLike is a minimal implementation of the (unnormalised)
    Gaussian likelihood function, with bias-correction.

    Attributes
    ----------
    data : numpy.ndarray
        The Gaussian distributed data.
    cov : numpy.ndarray
        The covariance of the Gaussian likelihood function.
    cov_inverse : numpy.ndarray
        The inverse of the covariance.
    bias : numpy.ndarray
        The mean of the bias.
    cov_bias
        The covariance of the bias.

    Methods
    ----------
    set_bias(bias, covariance_bias)
        Set the bias of the likelihood function.
    loglike(x)
        Returns the log-likelihood of the input array x, correcting for the bias.
    loglike_custom_bias(x, bias)
        Returns the log-likelihood of the input array, correcting for a custom
        (one-shot) bias.
    """

    def __init__(self, data, covariance):
        """
        Parameters
        ----------
        data : numpy.ndarray
            The Gaussian distributed data.
        cov : numpy.ndarray
            The covariance of the Gaussian likelihood function.
        """
        # check if covariance operator is a square numpy array.
        if not isinstance(covariance, np.ndarray):
            raise TypeError("Covariance must be a 2-D numpy array.")
        if covariance.ndim == 2:
            if not covariance.shape[0] == data.shape[0]:
                raise ValueError("Dimensions of data and covariance do not match.")
            if not covariance.shape[0] == covariance.shape[1]:
                raise ValueError("Covariance must be an NxN array.")
        else:
            raise TypeError("Covariance must be a 2-D numpy array.")

        super().__init__(data, covariance)

        # set the initial bias.
        self.bias = np.zeros(self.data.shape[0])

    def set_bias(self, mean_bias, covariance_bias):
        """
        Parameters
        ----------
        mean_bias : numpy.ndarray
            The mean of the bias.
        covariance_bias : numpy.ndarray
            The covariance of the bias.
        """
        # set the bias and the covariance of the bias.
        self.bias = mean_bias
        self.cov_bias = covariance_bias

        # precompute the inverse.
        if np.all(self.cov_bias < 1e-9):
            pass
        else:
            self.cov_inverse = np.linalg.inv(self.cov + self.cov_bias)

    def loglike(self, x):
        """
        Parameters
        ----------
        x : numpy.ndarray
            Input array, to be evaluated by the likelihood function.

        Returns
        ----------
        float
            The bias-corrected log-likelihood of the input array.
        """

        # compute the unnormalised likelihood, with additional terms for
        # offset, scaling, and rotation.
        return -0.5 * np.linalg.multi_dot(
            (
                (x + self.bias - self.data).T,
                self.cov_inverse,
                (x + self.bias - self.data),
            )
        )

    def loglike_custom_bias(self, x, bias):
        """
        Parameters
        ----------
        x : numpy.ndarray
            Input array, to be evaluated by the likelihood function.
        bias : numpy.ndarray
            A custom bias to add to the input array.

        Returns
        ----------
        float
            The custom bias-corrected log-likelihood of the input array.
        """

        # compute the unnormalised likelihood, with additional terms for
        # offset, scaling, and rotation.
        return -0.5 * np.linalg.multi_dot(
            ((x + bias - self.data).T, self.cov_inverse, (x + bias - self.data))
        )

    def grad_loglike(self, x):
        return np.dot(self.cov_inverse, (self.data - (x + self.bias)))
