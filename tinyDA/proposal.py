# external imports
import warnings
import random
from copy import deepcopy

import numpy as np
from scipy.linalg import sqrtm
import scipy.stats as stats
from scipy.optimize import approx_fprime
from scipy.stats import gaussian_kde

# internal imports
from .utils import RecursiveSampleMoments, grad_log_p, grad_log_l


class Proposal:
    """Base proposal.

    Only used for inheritance.
    """

    is_symmetric = False

    def __init__(self):
        pass

    def setup_proposal(self, **kwargs):
        pass

    def adapt(self, **kwargs):
        pass

    def make_proposal(self, link):
        pass

    def get_acceptance(self, proposal_link, previous_link):
        pass

    def get_q(self, x_link, y_link):
        pass


class IndependenceSampler(Proposal):

    """Independence sampler using a proposal distribution q(x).

    Attributes
    ----------
    q : scipy.stats.rv_continuous
        A probability distribution to draw independent samples from. Usually
        a scipy distribution, but the only requirement is that it has .rvs()
        and .logpdf() methods.

    Methods
    ----------
    setup_proposal(**kwargs)
        This proposal does not require any setup.
    adapt(**kwargs)
        This proposal is not adaptive.
    make_proposal(link)
        Generates an independent proposal from q.
    get_acceptance(proposal_link, previous_link)
        Computes the acceptance probability given a proposal link and the
        previous link.
    get_q(x_link, y_link)
        Computes the transition probability from x_link to y_link.
    """

    def __init__(self, q):
        """
        Parameters
        ----------
        q : scipy.stats.rv_continuous
            A probability distribution to draw independent samples from.
            Usually a scipy distribution, but the only requirement is that
            it has .rvs() and .logpdf() methods.
        """

        # set the proposal distribution.
        self.q = q

        # test that the proposal distribution has the required methods.
        try:
            self.q.logpdf(self.q.rvs(1))
        except AttributeError:
            raise

    def setup_proposal(self, **kwargs):
        pass

    def adapt(self, **kwargs):
        pass

    def make_proposal(self, link):
        # draw a random sample from the proposal distribution.
        return self.q.rvs(1).flatten()

    def get_acceptance(self, proposal_link, previous_link):
        q_proposal = self.get_q(None, proposal_link)
        q_previous = self.get_q(None, previous_link)

        return np.exp(
            proposal_link.posterior - previous_link.posterior + q_previous - q_proposal
        )

    def get_q(self, x_link, y_link):
        # get the transition probability.:
        return self.q.logpdf(y_link.parameters)


class GaussianRandomWalk(Proposal):

    """Standard Random Walk Metropolis Hastings proposal.

    Attributes
    ----------
    C : np.ndarray
        The covariance matrix of the proposal distribution.
    d : int
        The dimension of the target distribution.
    scaling : float
        The global scaling of the proposal.
    adaptive : bool
        Whether to adapt the global scaling of the proposal.
    gamma : float
        The adaptivity coefficient for global adaptive scaling.
    period : int
        How often to adapt the global scaling.
    k : int
        How many times the proposal has been adapted.
    t : int
        How many times the adapt method has been called.

    Methods
    ----------
    setup_proposal(**kwargs)
        This proposal does not require any setup.
    adapt(**kwargs)
        If adaptive=True, the proposal will adapt the global scaling.
    make_proposal(link)
        Generates a random walk proposal from the input link using the
        proposal covariance.
    get_acceptance(proposal_link, previous_link)
        Computes the acceptance probability given a proposal link and the
        previous link.
    """

    is_symmetric = True
    alpha_star = 0.24

    def __init__(self, C, scaling=1, adaptive=False, gamma=1.01, period=100):
        """
        Parameters
        ----------
        C : np.ndarray
            The covariance matrix of the proposal distribution.
        scaling : float, optional
            The global scaling of the proposal. Default is 1.
        adaptive : bool, optional
            Whether to adapt the global scaling of the proposal. Default is
            False.
        gamma : float, optional
            The adaptivity coefficient for the global adaptive scaling.
            Default is 1.01.
        period : int, optional
            How often to adapt the global scaling. Default is 100.
        """

        # check if covariance operator is a square numpy array.
        if not isinstance(C, np.ndarray):
            raise TypeError("C must be a numpy array")
        elif C.ndim == 1:
            if not C.shape[0] == 1:
                raise ValueError("C must be an NxN array")
        elif not C.shape[0] == C.shape[1]:
            raise ValueError("C must be an NxN array")

        # set the covariance operator
        self.C = C

        # extract the dimensionality.
        self.d = self.C.shape[0]

        # set the distribution mean to zero.
        self._mean = np.zeros(self.d)

        # set the scaling.
        self.scaling = scaling

        # set adaptivity.
        self.adaptive = adaptive

        # if adaptive, set some adaptivity parameters
        if self.adaptive:
            # adaptivity scaling.
            self.gamma = gamma
            # adaptivity period (delay between adapting)
            self.period = period
            # initialise adaptivity counter for diminishing adaptivity.
            self.k = 0

        # initialise counter of how many times, adapt() has been called..
        self.t = 0

    def setup_proposal(self, **kwargs):
        pass

    def adapt(self, **kwargs):
        self.t += 1

        # if adaptive, run the adaptivity routines.
        if self.adaptive:
            # make sure the periodicity is respected
            if self.t % self.period == 0:
                # compute the acceptance rate during the previous period.
                acceptance_rate = np.mean(kwargs["accepted"][-self.period :])
                # set the scaling so that the acceptance rate will converge to 0.24.
                self.scaling = np.exp(
                    np.log(self.scaling)
                    + self.gamma**-self.k * (acceptance_rate - self.alpha_star)
                )
                # increase adaptivity counter for diminishing adaptivity.
                self.k += 1
        else:
            pass

    def make_proposal(self, link):
        # make a Gaussian RWMH proposal.
        return link.parameters + self.scaling * np.random.multivariate_normal(
            self._mean, self.C
        )

    def get_acceptance(self, proposal_link, previous_link):
        if np.isnan(proposal_link.posterior):
            return 0
        else:
            # get the acceptance probability.
            return np.exp(proposal_link.posterior - previous_link.posterior)


class CrankNicolson(GaussianRandomWalk):

    """Preconditioned Crank Nicolson proposal. To use this proposal, the prior
    must be of the type scipy.stats.multivariate_normal.

    Attributes
    ----------
    C : np.ndarray
        The covariance matrix of the proposal distribution. This is set
        to the prior covariance, as required by pCN.
    d : int
        The dimension of the target distribution.
    scaling : float
        The global scaling ("beta") of the proposal.
    adaptive : bool
        Whether to adapt the global scaling of the proposal.
    gamma : float
        The adaptivity coefficient for the global adaptive scaling.
    period : int
        How often to adapt the global scaling.
    k : int
        How many times the proposal has been adapted.
    t : int
        How many times the adapt method has been called.

    Methods
    ----------
    setup_proposal(**kwargs)
        Sets the proposal covariance to the prior covariance, as required
        for pCN.
    adapt(**kwargs)
        If adaptive=True, the proposal will adapt the global scaling.
    make_proposal(link)
        Generates a pCN proposal from the input link using the proposal (prior)
        covariance.
    get_acceptance(proposal_link, previous_link)
        Computes the acceptance probability given a proposal link and the
        previous link.
    """

    is_symmetric = False

    def __init__(self, scaling=0.1, adaptive=False, gamma=1.01, period=100):
        """
        Parameters
        ----------
        scaling : float, optional
            The global scaling ("beta") of the proposal. Default is 0.1.
        adaptive : bool, optional
            Whether to adapt the global scaling of the proposal. Default is
            False.
        gamma : float, optional
            The adaptivity coefficient for the global adaptive scaling. Default
            is 1.01.
        period : int, optional
            How often to adapt the global scaling. Default is 100.
        """

        # set the scaling.
        self.scaling = scaling

        # set adaptivity.
        self.adaptive = adaptive

        # if adaptive, set some adaptivity parameters
        if self.adaptive:
            # adaptivity scaling.
            self.gamma = gamma
            # adaptivity period (delay between adapting)
            self.period = period
            # initialise adaptivity counter for diminishing adaptivity.
            self.k = 0

        # initialise counter of how many times, adapt() has been called..
        self.t = 0

    def setup_proposal(self, **kwargs):
        # set the covariance operator
        try:
            self.C = kwargs["posterior"].prior.cov
        except AttributeError:
            self.C = kwargs["posterior"].prior.cov_object.covariance

        # extract the dimensionality.
        self.d = self.C.shape[0]

        # set the distribution mean to zero.
        self._mean = np.zeros(self.d)

    def make_proposal(self, link):
        # make a pCN proposal.
        return np.sqrt(
            1 - self.scaling**2
        ) * link.parameters + self.scaling * np.random.multivariate_normal(
            self._mean, self.C
        )

    def get_acceptance(self, proposal_link, previous_link):
        if np.isnan(proposal_link.posterior):
            return 0
        else:
            # get the acceptance probability.
            return np.exp(proposal_link.likelihood - previous_link.likelihood)

    def get_q(self, x_link, y_link):
        return stats.multivariate_normal.logpdf(
            y_link.parameters,
            mean=np.sqrt(1 - self.scaling**2) * x_link.parameters,
            cov=self.scaling**2 * self.C,
        )


class AdaptiveMetropolis(GaussianRandomWalk):

    """Adaptive Metropolis proposal, according to Haario et al. (2001)

    Attributes
    ----------
    C : np.ndarray
        The covariance matrix of the proposal distribution.
    d : int
        The dimension of the target distribution.
    scaling : float
        The global scaling of the proposal.
    sd : float
        The AM scaling parameter.
    epsilon : float
        Parameter to prevent C from becoming singular.
    t0 : int
        When to start the adapting covariance matrix.
    adaptive : bool
        Whether to adapt the global scaling of the proposal.
    gamma : float
        The adaptivity coefficient for the global adaptive scaling.
    period : int
        How often to adapt the global scaling.
    k : int
        How many times the proposal has been globally adapted.
    t : int
        How many times the adapt method has been called.

    Methods
    ----------
    setup_proposal(**kwargs)
        A tinyDA.RecursiveSampleMoments is initialised to adapt the proposal
        covariance.
    adapt(**kwargs)
        The tinyDA.RecursiveSampleMoments is adapted using the latest sample.
        If adaptive=True, the proposal will also adapt the global scaling.
    make_proposal(link)
        Generates an Adaptive Metropolis proposal from the input link using
        the proposal covariance.
    get_acceptance(proposal_link, previous_link)
        Computes the acceptance probability given a proposal link and the
        previous link.
    """

    def __init__(
        self, C0, sd=None, epsilon=1e-6, t0=0, period=100, adaptive=False, gamma=1.01
    ):
        """
        Parameters
        ----------
        C0 : np.ndarray
            The initial covariance matrix of the proposal distribution.
        sd : None or float, optional
            The AM scaling parameter. Default is None (compute from target
            dimensionality).
        epsilon : float, optional
            Parameter to prevent C from becoming singular. Must be small.
            Default is 1e-6.
        t0 : int, optional
            When to start adapting the covariance matrix. Default is 0 (start
            immediately).
        period : int, optional
            How often to adapt. Default is 100.
        adaptive : bool, optional
            Whether to adapt the global scaling of the proposal. Default is
            False.
        gamma : float, optional
            The adaptivity coefficient for the global adaptive scaling.
            Default is 1.01.
        """

        # check if covariance operator is a square numpy array.
        if not isinstance(C0, np.ndarray):
            raise TypeError("C0 must be a numpy array")
        elif C0.ndim == 1:
            if not C0.shape[0] == 1:
                raise ValueError("C0 must be an NxN array")
        elif not C0.shape[0] == C0.shape[1]:
            raise ValueError("C0 must be an NxN array")

        # set the initial covariance operator.
        self.C = C0

        # extract the dimensionality.
        self.d = self.C.shape[0]

        # set a zero mean for the random draw.
        self._mean = np.zeros(self.d)

        # set the scaling factor.
        self.scaling = 1

        # Set the scaling parameter for Diminishing Adaptation.
        if sd is not None:
            self.sd = sd
        else:
            self.sd = min(1, 2.4**2 / self.d)

        # Set epsilon to avoid degeneracy.
        self.epsilon = epsilon

        # set the beginning of adaptation (rigidness of initial covariance).
        self.t0 = t0

        # Set the update period.
        self.period = period

        # set adaptivity.
        self.adaptive = adaptive

        # if adaptive, set some adaptivity parameters
        if self.adaptive:
            # adaptivity scaling.
            self.gamma = gamma
            # initialise adaptivity counter for diminishing adaptivity.
            self.k = 0

        # set a counter of how many times, adapt() has been called.
        self.t = 0

    def setup_proposal(self, **kwargs):
        # initialise the sampling moments, which will compute the
        # adaptive covariance operator.
        self.AM_recursor = RecursiveSampleMoments(
            kwargs["parameters"],
            np.zeros((self.d, self.d)),
            sd=self.sd,
            epsilon=self.epsilon,
        )

    def adapt(self, **kwargs):
        super().adapt(**kwargs)

        # AM is adaptive per definition. update the RecursiveSampleMoments
        # with the given parameters.
        self.AM_recursor.update(kwargs["parameters"])

        if self.t >= self.t0 and self.t % self.period == 0:
            self.C = self.AM_recursor.get_sigma()
        else:
            pass


class OperatorWeightedCrankNicolson(CrankNicolson):

    """Operator-weighted preconditioned Crank-Nicolson proposal (Law 2014).

    Attributes
    ----------
    B : numpy.ndarray
        The scaling operator of the proposal distribution.
    scaling : float
        The global scaling of the proposal.
    t0 : int
        When to start adapting the covariance matrix.
    adaptive : bool
        Whether to adapt the global scaling of the proposal.
    gamma : float
        The adaptivity coefficient for the global adaptive scaling.
    period : int
        How often to adapt the global scaling.
    k : int
        How many times the proposal has been globally adapted.
    t : int
        How many times the adapt method has been called.

    Methods
    ----------
    setup_proposal(**kwargs)
        Setup the proposal to allow adaption according to Hu and Yao (2016).
    adapt(**kwargs)
        Use the latest sample to update the adaptive pCN operator.
        If adaptive=True, the proposal will also adapt the global scaling.
    make_proposal(link)
        Generates an adaptive pCN proposal from the input link using the
        proposal covariance.
    get_acceptance(proposal_link, previous_link)
        Computes the acceptance probability given a proposal link and the
        previous link.
    """

    def __init__(self, B, scaling=1.0, adaptive=False, gamma=1.01, period=100):
        """
        Parameters
        ----------
        B : numpy.ndarray
            The scaling operator of the proposal distribution.
        scaling : float, optional
            The global scaling of the proposal. Default is 1.0.
        adaptive : bool, optional
            Whether to adapt the global scaling of the proposal. Default is
            False.
        gamma : float, optional
            The adaptivity coefficient for the global adaptive scaling.
            Default is 1.01.
        period : int, optional
            How often to adapt. Default is 100.
        """

        # set the scaling operator
        self.B = B

        super().__init__(scaling, adaptive, gamma, period)

    def setup_proposal(self, **kwargs):
        super().setup_proposal(**kwargs)

        self.state_operator = np.real(sqrtm(np.eye(self.d) - self.scaling * self.B))
        self.noise_operator = np.real(sqrtm(self.scaling * self.B))

    def adapt(self, **kwargs):
        super().adapt(**kwargs)

        if self.adaptive:
            # make sure the periodicity is respected
            if self.t % self.period == 0:
                self.state_operator = np.real(
                    sqrtm(np.eye(self.d) - self.scaling * self.B)
                )
                self.noise_operator = np.real(sqrtm(self.scaling * self.B))

    def make_proposal(self, link):
        # only use the adaptive proposal, if the initial time has passed.

        # make a proposal
        return np.dot(self.state_operator, link.parameters) + np.dot(
            self.noise_operator, np.random.multivariate_normal(self._mean, self.C)
        )

    def get_q(self, x_link, y_link):
        return stats.multivariate_normal.logpdf(
            y_link.parameters,
            mean=np.dot(self.state_operator, x_link.parameters),
            cov=np.dot(self.scaling * self.B, self.C),
        )


class DREAMZ(GaussianRandomWalk):

    """Dream(Z) proposal, similar to the DREAM(ZS) algorithm (see e.g. Vrugt
    2016).

    Attributes
    ----------
    M0 : int
        Size of the initial archive.
    d : int
        The dimension of the target distribution.
    scaling : float
        The global scaling of the proposal.
    delta : int
        Number of sample pairs from the archive to use to compute the jumping
        direction.
    b : float
        Upper and lower bound for the uniform pertubation distribution,
        i.e. e ~ U(-b,b).
    b_star : float
        Scale for the Gaussian pertubation distribution, i.e.
        epsilon ~ N(0, b_star).
    Z_method : str
        How to draw the initial archive.
        Can be 'random' for simple random sampling or 'lhs' for latin hypercube
        sampling.
    nCR : int
        Size of the crossover probability distribution.
    adaptive : bool
        Whether to adapt the global scaling and crossover distribution of
        the proposal.
    gamma : float
        The adaptivity coefficient for global adaptive scaling.
    period : int
        How often to adapt the global scaling and crossover probabilities.
    k : int
        How many times the proposal has been adapted.
    t : int
        How many times the adapt method has been called.

    Methods
    ----------
    setup_proposal(**kwargs)
        The proposal creates the initial archive from samples from the prior,
        using the specified Z-method. If set to "lhs" and pyDOE is not available,
        it will fall back to simple random sampling.
    adapt(**kwargs)
        If adaptive=True, the proposal will adapt the global scaling and the
        crossover distribution.
    make_proposal(link)
        Generates a SingleDreamZ proposal.
    get_acceptance(proposal_link, previous_link)
        Computes the acceptance probability given a proposal link and the
        previous link.
    """

    def __init__(
        self,
        M0,
        delta=1,
        b=5e-2,
        b_star=1e-6,
        Z_method="random",
        nCR=3,
        adaptive=False,
        gamma=1.01,
        period=100,
    ):
        """
        Parameters
        ----------
        M0 : np.ndarray
            Size of the initial archive.
        delta : int, optional
            Number of sample pairs from the archive to use to compute the
            jumping direction.
            Default is 1.
        b : float, optional
            Upper and lower bound for the uniform pertubation distribution,
            i.e. e ~ U(-b,b).
            Default is 5e-2.
        b_star : float, optional
            Scale for the Gaussian pertubation distribution,
            i.e. epsilon ~ N(0, b_star).
            Must be small. Default is 1e-6.
        Z_method : str, optional
            How to draw the initial archive.
            Can be 'random' for simple random sampling or 'lhs' for latin
            hypercube sampling.
            Default is 'random'.
        nCR : int, optional
            Size of the crossover probability distribution. Default is 3.
        adaptive : bool, optional
            Whether to adapt the global scaling of the proposal and crossover
            distribution.
            Default is False.
        gamma : float, optional
            The adaptivity coefficient for global adaptive scaling. Default
            is 1.01.
        period : int, optional
            How often to adapt the global scaling and crossover distribution.
            Default is 100.
        """

        # set initial archive size.
        self.M = M0

        # set global scaling
        self.scaling = 1

        # set DREAM parameters
        self.delta = delta
        self.b = b
        self.b_star = b_star

        # Set the method to create the initial archive.
        self.Z_method = Z_method

        # set adaptivity.
        self.adaptive = adaptive

        # set crossover distribution.
        self.nCR = nCR
        self.mCR = None
        self.pCR = np.array(self.nCR * [1 / self.nCR])

        # if adaptive, set some adaptivity parameters
        if self.adaptive:
            # adaptivity scaling.
            self.gamma = gamma
            # adaptivity period (delay between adapting)
            self.period = period
            # initialise adaptivity counter for diminishing adaptivity.
            self.k = 0

        self.t = 0

    def setup_proposal(self, **kwargs):
        prior = kwargs["posterior"].prior

        # get the dimension and the initial scaling.
        self.d = kwargs["posterior"].prior.rvs().size

        # these adaptivity parameters need to be mutable, so are only
        # initialised at setup.
        if self.adaptive:
            # DREAM-specific adaptivity
            self.LCR = np.zeros(self.nCR)
            self.DeltaCR = np.ones(self.nCR)

        # draw initial archive with latin hypercube sampling.
        if self.Z_method == "lhs":
            try:
                lhs = stats.qmc.LatinHypercube(d=self.d)
                self.Z = lhs.random(n=self.M)
                self.Z = prior.ppf(self.Z)
                return

            except AttributeError:
                # if the prior is a multivariate_normal, it will not have a .ppf-method.
                if isinstance(prior, stats._multivariate.multivariate_normal_frozen):
                    try:
                        var = np.diag(prior.cov)
                    except AttributeError:
                        var = np.diag(prior.cov_object.covariance)

                    # instead draw samples from independent normals, according to the prior means and variances.
                    for i in range(self.d):
                        self.Z[:, i] = stats.norm(
                            loc=prior.mean[i], scale=np.sqrt(var[i])
                        ).ppf(self.Z[:, i])
                    return

                else:
                    # if the prior does not have a .ppf-method, and it's not a multivariate gaussian, fall back on simple random sampling.
                    warnings.warn(
                        " Prior does not have .ppf method. Falling back on default Z-sampling method: 'random'.\n"
                    )
                    pass

        # do simple random sampling from the prior.
        self.Z = prior.rvs(self.M)

    def adapt(self, **kwargs):
        super().adapt(**kwargs)

        # extend the archive.
        self.Z = np.vstack((self.Z, kwargs["parameters"]))
        self.M = self.Z.shape[0]

        # adaptivity
        if self.adaptive and self.t % self.period == 0:
            # compute new multinomial distribution according to the normalised jumping distance.
            jumping_distance = kwargs["parameters"] - kwargs["parameters_previous"]
            self.DeltaCR[self.mCR] = (
                self.DeltaCR[self.mCR]
                + (jumping_distance**2 / np.var(self.Z, axis=0)).sum()
            )
            self.LCR[self.mCR] = self.LCR[self.mCR] + 1

            if np.all(self.LCR > 0):
                DeltaCR_mean = self.DeltaCR / self.LCR
                self.pCR = DeltaCR_mean / DeltaCR_mean.sum()

    def make_proposal(self, link):
        # initialise the jump vectors.
        Z_r1 = np.zeros(self.d)
        Z_r2 = np.zeros(self.d)

        # get jump vector components.
        for i in range(self.delta):
            r1, r2 = np.random.choice(self.M, 2, replace=False)
            Z_r1 += self.Z[r1, :]
            Z_r2 += self.Z[r2, :]

        # randomly choose crossover probability.
        self.mCR = np.random.choice(self.nCR, p=self.pCR)
        CR = (self.mCR + 1) / self.nCR

        # set up the subspace indicator, deciding which dimensions to pertubate.
        subspace_indicator = np.zeros(self.d)
        subspace_draw = np.random.uniform(size=self.d)
        subspace_indicator[subspace_draw < CR] = 1

        # if no dimensions were chosen, pick one a random.
        if subspace_indicator.sum() == 0:
            subspace_indicator[np.random.choice(self.d)] = 1

        # compute the optimal scaling.
        gamma_DREAM = (
            self.scaling * 2.38 / np.sqrt(2 * self.delta * subspace_indicator.sum())
        )

        # get the random scalings and gaussian pertubation.
        e = np.random.uniform(-self.b, self.b, size=self.d)
        epsilon = np.random.normal(0, self.b_star, size=self.d)

        return link.parameters + subspace_indicator * (
            (np.ones(self.d) + e) * gamma_DREAM * (Z_r1 - Z_r2) + epsilon
        )


def SingleDreamZ(*args, **kwargs):
    """Deprecation dummy."""
    warnings.warn(" SingleDreamZ has been deprecated. Please use DREAMZ.")
    return DREAMZ(*args, **kwargs)


class MALA(Proposal):
    """
    Metropolis-Adjusted Langevin Algorithm (MALA) proposal. If the model
    does not implement a "gradient" method, the posterior gradient will be
    approximated using finite differences.

    Attributes
    ----------
    scaling : float
        The global scaling ("sigma") of the proposal.
    adaptive : bool
        Whether to adapt the global scaling of the proposal.
    gamma : float
        The adaptivity coefficient for the global adaptive scaling.
    period : int
        How often to adapt the global scaling.
    k : int
        How many times the proposal has been adapted.
    t : int
        How many times the adapt method has been called.

    Methods
    ----------
    setup_proposal(**kwargs)
        Sets the dimension of the target and chooses the correct gradient
        function.
    adapt(**kwargs)
        If adaptive=True, the proposal will adapt the global scaling.
    make_proposal(link)
        Generates a MALA proposal from the input link.
    get_acceptance(proposal_link, previous_link)
        Computes the acceptance probability given a proposal link and the
        previous link.
    """

    alpha_star = 0.57

    def __init__(self, scaling=0.1, adaptive=False, gamma=1.01, period=100):
        """
        Parameters
        ----------
        scaling : float, optional
            The global scaling ("sigma") of the proposal. Default is 0.1.
        adaptive : bool, optional
            Whether to adapt the global scaling of the proposal. Default is
            False.
        gamma : float, optional
            The adaptivity coefficient for the global adaptive scaling. Default
            is 1.01.
        period : int, optional
            How often to adapt the global scaling. Default is 100.
        """

        # set the scaling.
        self.scaling = scaling

        # set adaptivity.
        self.adaptive = adaptive

        # if adaptive, set some adaptivity parameters
        if self.adaptive:
            # adaptivity scaling.
            self.gamma = gamma
            # adaptivity period (delay between adapting)
            self.period = period
            # initialise adaptivity counter for diminishing adaptivity.
            self.k = 0

        # initialise counter of how many times, adapt() has been called..
        self.t = 0

    def setup_proposal(self, **kwargs):
        self.posterior = kwargs["posterior"]

        # set the target dimension
        self.d = kwargs["posterior"].prior.rvs().size

        # set the gradient function. if the forward model has a gradient
        # method, use the exact gradient. othereise use a finite difference
        # approximation.
        model_gradient = getattr(self.posterior.model, "gradient", None)
        if callable(model_gradient):
            self.compute_gradient = self._compute_gradient
        else:
            self.compute_gradient = self._compute_gradient_approx

    def make_proposal(self, link):
        # compute the gradient for the input link, if it has not been
        # computed yet.
        if not hasattr(link, "gradient"):
            link.gradient = self.compute_gradient(link)

        # make a MALA proposal.
        return (
            link.parameters
            + 0.5 * self.scaling**2 * link.gradient
            + self.scaling * np.random.normal(size=self.d)
        )

    def get_acceptance(self, proposal_link, previous_link):
        # safety against NaN evaluations.
        if np.isnan(proposal_link.posterior):
            return 0

        # compute the gradient for the proposal, if it has not been
        # computed yet.
        if not hasattr(proposal_link, "gradient"):
            proposal_link.gradient = self.compute_gradient(proposal_link)

        # compute the forward and backwards transisitions probabilities.
        q_x_y = self.get_q(previous_link, proposal_link)
        q_y_x = self.get_q(proposal_link, previous_link)

        return np.exp(proposal_link.posterior - previous_link.posterior + q_x_y - q_y_x)

    def get_q(self, x_link, y_link):
        # get the MALA transition probability.
        return (
            -0.5
            / self.scaling**2
            * np.linalg.norm(
                x_link.parameters
                - y_link.parameters
                - 0.5 * self.scaling**2 * y_link.gradient
            )
            ** 2
        )

    def _compute_gradient(self, link):
        # get the gradient of the log-prior.
        grad_log_prior = grad_log_p(link.parameters, self.posterior.prior)
        # get the gradient of the likelihood function (not considering the forward model).
        grad_log_sensitivity = grad_log_l(link.model_output, self.posterior.likelihood)
        # get the gradient of the likelihood, i.e. np.dot(grad_log_sensitivity, model_jacobian).
        grad_log_likelikehood = self.posterior.model.gradient(
            link.parameters, grad_log_sensitivity
        )
        # return the gradient of the log-posterior.
        return grad_log_prior + grad_log_likelikehood

    def _compute_gradient_approx(self, link):
        # aproximate the gradient of the log-posterior using finite differences.
        grad_log_posterior = approx_fprime(
            link.parameters, lambda x: self.posterior.create_link(x).posterior
        )
        return grad_log_posterior


class KernelMALA(MALA):
    """
    Metropolis-Adjusted Langevin Algorithm (MALA) proposal using a kernel
    approximation for the gradient of the log-posterior. See e.g.
    Strathmann, H., Sejdinovic, D., Livingstone, S., Szabo, Z., & Gretton, A. (2015).
    Gradient-free Hamiltonian Monte Carlo with Efficient Kernel Exponential Families.
    Advances in Neural Information Processing Systems, 28.
    If no kernel is provided, scipy.stats.gaussian_kde will be used.

    Attributes
    ----------
    kernel : object
        The kernel used to approximate the gradient of the log-posterior.
    M : int
        The length of the sampling history used by the kernel.
    t0 : float
        When to start using the kernel gradient.
    scaling : float
        The global scaling ("sigma") of the proposal.
    adaptive : bool
        Whether to adapt the global scaling of the proposal.
    gamma : float
        The adaptivity coefficient for the global adaptive scaling.
    period : int
        How often to adapt the global scaling.
    k : int
        How many times the proposal has been adapted.
    t : int
        How many times the adapt method has been called.

    Methods
    ----------
    setup_proposal(**kwargs)
        Sets the dimension of the target and chooses the correct gradient
        function. Starts bukding the history.
    adapt(**kwargs)
        If adaptive=True, the proposal will adapt the global scaling.
    make_proposal(link)
        Generates a MALA proposal from the input link.
    get_acceptance(proposal_link, previous_link)
        Computes the acceptance probability given a proposal link and the
        previous link.
    """

    def __init__(
        self,
        kernel=None,
        M=1000,
        t0=1000,
        scaling=0.1,
        adaptive=False,
        gamma=1.01,
        period=100,
    ):
        """
        Parameters
        ----------
        kernel : object
            The kernel used to approximate the gradient of the log-posterior.
            Default is None, which will use scipy.stats.gaussian_kde as
            the kernel.
        M : int
            The length of the sampling history used by the kernel.
        t0 : float
            When to start using the kernel gradient.
        scaling : float
            The global scaling ("sigma") of the proposal.
        adaptive : bool
            Whether to adapt the global scaling of the proposal.
        gamma : float
            The adaptivity coefficient for the global adaptive scaling.
        period : int
            How often to adapt the global scaling and the kernel.
        """

        # set the kernel to scipy.stats.gaussian_kde if nothing is provided.
        if kernel is None:
            self._kernel = gaussian_kde
        else:
            self._kernel = kernel

        # set the history length.
        self.M = M
        # set the number of samples before the kernel will be used.
        self.t0 = t0

        super().__init__(scaling, adaptive, gamma, period)

    def setup_proposal(self, **kwargs):
        self.posterior = kwargs["posterior"]

        # set the target dimension.
        self.d = kwargs["posterior"].prior.rvs().size

        # initialise an array forthe sampling history (archive).
        self.Z = kwargs["parameters"]

    def adapt(self, **kwargs):
        super().adapt(**kwargs)

        # extend the archive.
        self.Z = np.vstack((self.Z, kwargs["parameters"]))

        # fit the kernel if it is time to do so.
        if self.t >= self.t0 and self.t % self.period == 0:
            self.kernel = self._kernel(self.Z[-self.M :, :].T)

    def compute_gradient(self, link):
        # get the gradient of the kernel.
        try:
            grad_log_posterior = approx_fprime(
                link.parameters, lambda x: self.kernel.logpdf(x)
            )
        # if there is no kernel yet, simply set the gradient to 0.
        except AttributeError:
            grad_log_posterior = 0
        return grad_log_posterior


class PoissonPointProposal(GaussianRandomWalk):

    """PoissonPointProposal can be seen as a special case of Reversible
    Jump MCMC. It is used to make moves for a PoissonPointProcess prior.

    Attributes
    ----------
    move_distribution : dict
        Specifies the (unnormalised) probability of making each of the
        named moves.
    moves : list
        A list of the functions used to make moves.
    probabilities : numpy.ndarray
        The probability of making each of the moves in the moves list.

    Methods
    ----------
    setup_proposal(**kwargs)
        This proposal extracts the prior in order to perturb points.
    adapt(**kwargs)
        This proposal is non-adaptive, so this method does nothing.
    make_proposal(link)
        Chooses a move according to the move probabilities and then
        executes that move.
    is_feasible(proposal)
        Convenience method that is triggered before a proposal is passed
        back to the sampler, allowing users to check for various geometric
        propoerties of the proposal before it is evaluated. By default
        returns True, so must be overwritten.
    create(x)
        Add a new point to the current state from the PoissonPointProcess
        prior.
    destroy(x)
        Remove a random point from the current state.
    move(x)
        Move a random point from the current state.
    shuffle(x)
        Randomly scramble the indices of all the points in the current state.
    swap(x)
        Randomly swap the indices of two points in the current state.
    perturb(x)
        Randomly perturb a single attribute of a point in the current state.
    """

    def __init__(
        self,
        move_distribution={
            "create": 1,
            "destroy": 1,
            "move": 1,
            "shuffle": 1,
            "swap": 1,
            "perturb": 1,
        },
    ):
        """
        Parameters
        ----------
        move_distribution : dict
            Specifies the (unnormalised) probability of making each of the
            named moves.
        """

        # internalise the dictionary of move probabilities.
        self.move_distribution = move_distribution

        # get the move functions and put them in a list.
        self.moves = [
            self.__getattribute__(move) for move in self.move_distribution.keys()
        ]
        # compute the normalised probabilities of each move.
        self.probabilities = np.array(
            [p for p in self.move_distribution.values()]
        ).astype(float)
        self.probabilities /= self.probabilities.sum()

        self.scaling = 0.1

    def setup_proposal(self, **kwargs):
        # link to the prior.
        self.prior = kwargs["posterior"].prior

    def adapt(self, **kwargs):
        pass

    def make_proposal(self, link):
        while True:
            try:
                # choose a move randomly.
                move = np.random.choice(self.moves, p=self.probabilities)
                # use the move function.
                proposal = move(link.parameters)
                # check if the move is feasible.
                if not self.is_feasible(proposal):
                    continue
            except ValueError:
                continue
            return proposal

    def is_feasible(self, proposal):
        return True

    def create(self, x):
        y = deepcopy(x)
        # draw a random index.
        idx = np.random.choice(range(len(x) + 1))
        # insert a new random point at the random index.
        y.insert(idx, self.prior._create_point())
        return y

    def destroy(self, x):
        y = deepcopy(x)
        # draw a random index.
        idx = np.random.choice(range(len(x)))
        # delete the point at the chosen index.
        del y[idx]
        return y

    def move(self, x):
        y = deepcopy(x)
        # draw a random index.
        idx = np.random.choice(range(len(x)))
        # delete the point at the chosen index.
        y[idx]["position"] += (
            np.random.choice([-1, 1]) * self.scaling * self.prior.domain_dist.rvs()
        )
        return y

    def shuffle(self, x):
        # randomly shuffle all the points.
        return random.sample(x, len(x))

    def swap(self, x):
        y = deepcopy(x)
        # iock two random indices.
        idx, idy = np.random.choice(range(len(x)), size=2, replace=False)
        # swap the points at the chosen indices.
        y[idy], y[idx] = x[idx], x[idy]
        return y

    def perturb(self, x):
        y = deepcopy(x)
        # draw a random index.
        idx = np.random.choice(range(len(x)))
        # pick a random attribute.
        attr = np.random.choice(list(self.prior.attributes.keys()))
        # get a new random draw for that attribute from the prior.
        y[idx][attr] += (
            np.random.choice([-1, 1]) * self.scaling * self.prior.attributes[attr].rvs()
        )
        return y


class MLDA(Proposal):

    """MLDAChain is a Multilevel Delayed Acceptance sampler. It takes a list of
    posteriors of increasing level as input, as well as a proposal, which
    applies to the coarsest level only.

    Attributes
    ----------
    posterior : tinyDA.Posterior
        The current level posterior responsible for communation between
        prior, likelihood and model. It also generates instances of
        tinyDA.Link (sample objects).
    level : int
        The (numeric) level of the current level sampler.
    proposal : tinyDA.MLDA or tinyDA.Proposal
        Transition kernel for the next-coarser MCMC proposals.
    subsampling_rate : int
        The subsampling rate for the next-coarser chain.
    initial_parameters : numpy.ndarray
        Starting point for the MCMC sampler
    chain : list
        Samples ("Links") in the current level MCMC chain.
    accepted : list
        List of bool, signifying whether a proposal was accepted or not.
    is_local : list
        List of bool, signifying whether a link was created on the current
        level or by correction according to the next-finer level.
    adaptive_error_model : str or None
        The adaptive error model used, see e.g. Cui et al. (2019).
    bias : tinaDA.RecursiveSampleMoments
        A recursive Gaussian error model that computes the sample moments
        of the next-coarser bias.

    Methods
    ----------
    setup_adaptive_error_model(biases)
        Sets up the adaptive error model and passes the biases to the
        next-coarser level.
    align_chain(parameters, accepted)
        Makes sure the first link of the current level subchain is aligned
        with the next-finer level.
    make_mlda_proposal(subsampling_rate)
        Generates a proposal by creating a subchain of length subsampling_rate
        using MLDA and passing the final link to the next-finer level.
    make_base_proposal(subsampling_rate)
        Generates a proposal by creating a subchain of length subsampling_rate
        using Metropolis-Hastings and passing the final link to the
        next-finer level.
    get_acceptance(proposal_link, previous_link)
        Computes the MLDA acceptance probability given a proposal link
        and the previous link.
    """

    is_symmetric = True

    def __init__(
        self,
        posteriors,
        proposal,
        subsampling_rates,
        initial_parameters,
        adaptive_error_model,
        store_coarse_chain,
    ):
        """
        Parameters
        ----------
        posteriors : list
            List of instances of tinyDA.Posterior, in increasing order.
        proposal : tinyDA.Proposal
            Transition kernel for coarsest MCMC proposals.
        subsampling_rates : list
            List of subsampling rates. It must have length
            len(posteriors) - 1, in increasing order.
        initial_parameters : numpy.ndarray, optional
            Starting point for the MCMC sampler, default is None (random
            draw from prior).
        adaptive_error_model : str or None, optional
            The adaptive error model, see e.g. Cui et al. (2019). Default
            is None (no error model), options are 'state-independent' or
            'state-dependent'. If an error model is used, the likelihood
            MUST have a set_bias() method, use e.g. tinyDA.AdaptiveLogLike.
        """

        # internalise the current level posterior and set the level.
        self.posterior = posteriors[-1]
        self.level = len(posteriors) - 1
        self.initial_parameters = initial_parameters

        # initialise lists for the links and the acceptance and localness histories.
        self.chain = []
        self.accepted = []
        self.is_local = []

        # create a link from the initial parameters and write to the histories.
        self.chain.append(self.posterior.create_link(self.initial_parameters))
        self.accepted.append(True)
        self.is_local.append(False)

        # set the adaptive error model as an attribute.
        self.adaptive_error_model = adaptive_error_model

        # set whether to store the coarse chain
        self.store_coarse_chain = store_coarse_chain

        # if this level is not the coarsest level.
        if self.level > 0:
            # internalise the subsampling rate.
            self.subsampling_rate = subsampling_rates[-1]

            # set MDLA as the proposal on the next-coarser level.
            self.proposal = MLDA(
                posteriors[:-1],
                proposal,
                subsampling_rates[:-1],
                self.initial_parameters,
                self.adaptive_error_model,
                self.store_coarse_chain,
            )

            # set the current level make_proposal method to MLDA.
            self.make_proposal = self.make_mlda_proposal

            # set up the adaptive error model.
            if self.adaptive_error_model is not None:
                # compute the difference between coarse and fine level.
                self.model_diff = (
                    self.chain[-1].model_output - self.proposal.chain[-1].model_output
                )

                # set up the state-independent adaptive error model.
                if self.adaptive_error_model == "state-independent":
                    # for the state-independent error model, the bias is
                    # RecursiveSampleMoments, and the corrector is the mean
                    # of all sampled differences.
                    self.bias = RecursiveSampleMoments(
                        self.model_diff,
                        np.zeros((self.model_diff.shape[0], self.model_diff.shape[0])),
                    )

                # state-dependent error model has not been implemented, since
                # it may not be ergodic.
                elif self.adaptive_error_model == "state-dependent":
                    pass

        # if the current level is the coarsest one.
        elif self.level == 0:
            # use the coarsest level proposal.
            self.proposal = proposal

            # set up the proposal.
            self.proposal.setup_proposal(
                parameters=self.initial_parameters, posterior=self.posterior
            )

            # set the current level make_proposal method to Metropolis-Hastings.
            self.make_proposal = self.make_base_proposal

    def setup_adaptive_error_model(self, biases):
        """
        Parameters
        ----------
        biases : list
            List of instances of tinyDA.RecursiveSampleMoments.
        """

        # add the current level bias to the list.
        self.biases = [self.bias] + biases

        # compute the total bias on the current level.
        mu_bias = np.sum([bias.get_mu() for bias in self.biases], axis=0)
        sigma_bias = np.sum([bias.get_sigma() for bias in self.biases], axis=0)

        # set the bias on the next-coarser level.
        self.proposal.posterior.likelihood.set_bias(mu_bias, sigma_bias)

        # update the first coarser link with the adaptive error model.
        self.proposal.chain[-1] = self.proposal.posterior.update_link(
            self.proposal.chain[-1]
        )

        # pass the list of biases on to the next level.
        if self.level > 1:
            self.proposal.setup_adaptive_error_model(self.biases)

    def align_chain(self, parameters, accepted):
        """
        Parameters
        ----------
        parameters : numpy.ndarray
            The latest accepted parameters on the next-finer level.
        accepted : bool
            Whether the passed parameters were accepted on the next-higher
            level.
        """

        # append the latest link on the current level matching the parameters.
        self.chain.append(
            next(filter(lambda link: link.parameters is parameters, self.chain[::-1]))
        )

        # add the acceptance bool to the history.
        self.accepted.append(accepted)

        # the appended link is not local.
        self.is_local.append(False)

        # perpetuate the correction downward in the model hierachy.
        if self.level > 0:
            self.proposal.align_chain(parameters, accepted)

    def _reset_chain(self):
        # remove everything except the latest coarse link, if the coarse
        # chain shouldn't be stored.
        self.chain = [self.chain[-1]]
        if self.level > 0:
            self.proposal._reset_chain()

    def make_mlda_proposal(self, subsampling_rate):
        """
        Parameters
        ----------
        subsampling rate : int
            The number of samples drawn in the subchain.
        """

        # iterate through the subsamples.
        for i in range(subsampling_rate):
            # create a proposal from the next-lower level,
            proposal = self.proposal.make_proposal(self.subsampling_rate)

            # if there were no acceptances on the next-lower level, repeat previous sample.
            if sum(self.proposal.accepted[-self.subsampling_rate :]) == 0:
                self.chain.append(self.chain[-1])
                self.accepted.append(False)
                self.is_local.append(True)

            # otherwise, evaluate the model.
            else:
                # create a link from that proposal.
                proposal_link = self.posterior.create_link(proposal)

                # compute the MLDA acceptance probability..
                alpha = self.proposal.get_acceptance(
                    proposal_link,
                    self.chain[-1],
                    self.proposal.chain[-1],
                    self.proposal.chain[-(self.subsampling_rate + 1)],
                )

                # perform Metropolis adjustment.
                if np.random.random() < alpha:
                    self.chain.append(proposal_link)
                    self.accepted.append(True)
                    self.is_local.append(True)
                else:
                    self.chain.append(self.chain[-1])
                    self.accepted.append(False)
                    self.is_local.append(True)

            # make sure the lower level chains are aligned with the current state.
            self.proposal.align_chain(self.chain[-1].parameters, self.accepted[-1])

            # apply the adaptive error model.
            if self.adaptive_error_model is not None:
                # compute the difference between coarse and fine level.
                if self.accepted[-1]:
                    self.model_diff = (
                        self.chain[-1].model_output
                        - self.proposal.chain[-1].model_output
                    )

                if self.adaptive_error_model == "state-independent":
                    # for the state-independent error model, the bias is
                    # RecursiveSampleMoments, and the corrector is the mean
                    # of all sampled differences.
                    self.bias.update(self.model_diff)

                    # compute the entire bias correction.
                    mu_bias = np.sum([bias.get_mu() for bias in self.biases], axis=0)
                    sigma_bias = np.sum(
                        [bias.get_sigma() for bias in self.biases], axis=0
                    )

                    # update the next-coarser likelihood with the bias.
                    self.proposal.posterior.likelihood.set_bias(mu_bias, sigma_bias)

                # state-dependent error model has not been implemented.
                elif self.adaptive_error_model == "state-dependent":
                    pass

                # update the latest link on the next-coarser level with the updated error model.
                self.proposal.chain[-1] = self.proposal.posterior.update_link(
                    self.proposal.chain[-1]
                )

        # return the latest link.
        return self.chain[-1].parameters

    def make_base_proposal(self, subsampling_rate):
        # iterate through the subsamples.
        for i in range(subsampling_rate):
            # draw a new proposal, given the previous parameters.
            proposal = self.proposal.make_proposal(self.chain[-1])

            # create a link from that proposal.
            proposal_link = self.posterior.create_link(proposal)

            # compute the acceptance probability, according to the proposal.
            alpha = self.proposal.get_acceptance(proposal_link, self.chain[-1])

            # perform Metropolis adjustment.
            if np.random.random() < alpha:
                self.chain.append(proposal_link)
                self.accepted.append(True)
                self.is_local.append(True)
            else:
                self.chain.append(self.chain[-1])
                self.accepted.append(False)
                self.is_local.append(True)

            # adapt the proposal. if the proposal is set to non-adaptive,
            # this has no effect.
            self.proposal.adapt(
                parameters=self.chain[-1].parameters,
                parameters_previous=self.chain[-2].parameters,
                accepted=self.accepted,
            )
        # return the latest link.
        return self.chain[-1].parameters

    def get_acceptance(
        self, proposal_link, previous_link, proposal_link_below, previous_link_below
    ):
        # get the MDLA acceptance probability.
        return np.exp(
            proposal_link.posterior
            - previous_link.posterior
            + previous_link_below.posterior
            - proposal_link_below.posterior
        )
