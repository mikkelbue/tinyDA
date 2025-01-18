import ray
import warnings

from itertools import compress
import numpy as np
from scipy.special import logsumexp

from .chain import Chain, DAChain, MLDAChain
from .proposal import *


class ParallelChain:

    """ParallelChain creates n_chains instances of tinyDA.Chain and runs the
    chains in parallel. It is initialsed with a Posterior (which holds the
    model and the distributions, and returns Links), and a proposal (transition
    kernel).

    Attributes
    ----------
    posterior : tinyDA.Posterior
        A posterior responsible for communation between prior, likelihood
        and model. It also generates instances of tinyDA.Link (sample objects).
    proposal : tinyDA.Proposal
        Transition kernel for MCMC proposals.
    n_chains : int
        Number of parallel chains.
    initial_parameters : list
        Starting points for the MCMC samplers
    remote_chains : list
        List of Ray actors, each running an independent MCMC sampler.

    Methods
    -------
    sample(iterations)
        Runs the MCMC for the specified number of iterations.
    """

    def __init__(self, posterior, proposal, n_chains=2, initial_parameters=None):
        """
        Parameters
        ----------
        posterior : tinyDA.Posterior
            A posterior responsible for communation between prior, likelihood
            and model. It also generates instances of tinyDA.Link (sample objects).
        proposal : tinyDA.Proposal
            Transition kernel for MCMC proposals.
        n_chains : int, optional
            Number of independent MCMC samplers. Default is 2.
        initial_parameters : list, optional
            Starting points for the MCMC samplers, default is None (random
            draws from prior).
        """

        # internalise the posterior and proposal.
        self.posterior = posterior
        self.proposal = proposal

        # set the number of parallel chains and initial parameters.
        self.n_chains = n_chains

        # set the initial parameters.
        self.initial_parameters = initial_parameters

        # initialise Ray.
        ray.init(ignore_reinit_error=True)

        # set up the parallel chains as Ray actors.
        self.remote_chains = [
            RemoteChain.remote(
                self.posterior, self.proposal[i], self.initial_parameters[i]
            )
            for i in range(self.n_chains)
        ]

    def sample(self, iterations, progressbar=False):
        """
        Parameters
        ----------
        iterations : int
            Number of MCMC samples to generate.
        progressbar : bool, optional
            Whether to draw a progressbar, default is False, since Ray
            and tqdm do not play very well together.
        """

        # initialise sampling on the chains and fetch the results.
        processes = [
            chain.sample.remote(iterations, progressbar) for chain in self.remote_chains
        ]
        self.chains = ray.get(processes)


class ParallelDAChain(ParallelChain):
    def __init__(
        self,
        posterior_coarse,
        posterior_fine,
        proposal,
        subchain_length=1,
        randomize_subchain_length=False,
        n_chains=2,
        initial_parameters=None,
        adaptive_error_model=None,
        store_coarse_chain=True,
    ):
        # internalise posteriors, proposal and subchain length.
        self.posterior_coarse = posterior_coarse
        self.posterior_fine = posterior_fine
        self.proposal = proposal
        self.subchain_length = subchain_length

        # set the number of parallel chains and initial parameters.
        self.n_chains = n_chains

        # set the initial parameters.
        self.initial_parameters = initial_parameters

        # set whether to randomize subchain length
        self.randomize_subchain_length = randomize_subchain_length

        # set the adaptive error model.
        self.adaptive_error_model = adaptive_error_model

        # whether to store the coarse chain.
        self.store_coarse_chain = store_coarse_chain

        # initialise Ray.
        ray.init(ignore_reinit_error=True)

        # set up the parallel DA chains as Ray actors.
        self.remote_chains = [
            RemoteDAChain.remote(
                self.posterior_coarse,
                self.posterior_fine,
                self.proposal[i],
                self.subchain_length,
                self.randomize_subchain_length,
                self.initial_parameters[i],
                self.adaptive_error_model,
                self.store_coarse_chain,
            )
            for i in range(self.n_chains)
        ]


class ParallelMLDAChain(ParallelChain):
    def __init__(
        self,
        posteriors,
        proposal,
        subchain_lengths=None,
        n_chains=2,
        initial_parameters=None,
        adaptive_error_model=None,
        store_coarse_chain=True,
    ):
        # internalise posteriors, proposal and subchain length.
        self.posteriors = posteriors
        self.proposal = proposal
        self.subchain_lengths = subchain_lengths

        # set the number of parallel chains and initial parameters.
        self.n_chains = n_chains

        # set the initial parameters.
        self.initial_parameters = initial_parameters

        # set the adaptive error model
        self.adaptive_error_model = adaptive_error_model

        # whether to store the coarse chain.
        self.store_coarse_chain = store_coarse_chain

        # initialise Ray.
        ray.init(ignore_reinit_error=True)

        # set up the parallel DA chains as Ray actors.
        self.remote_chains = [
            RemoteMLDAChain.remote(
                self.posteriors,
                self.proposal[i],
                self.subchain_lengths,
                self.initial_parameters[i],
                self.adaptive_error_model,
                self.store_coarse_chain,
            )
            for i in range(self.n_chains)
        ]


@ray.remote
class RemoteChain(Chain):
    def sample(self, iterations, progressbar):
        super().sample(iterations, progressbar)
        return self


@ray.remote
class RemoteDAChain(DAChain):
    def sample(self, iterations, progressbar):
        super().sample(iterations, progressbar)
        return self


@ray.remote
class RemoteMLDAChain(MLDAChain):
    def sample(self, iterations, progressbar):
        super().sample(iterations, progressbar)
        return self


class MultipleTry(Proposal):

    """Multiple-Try proposal (Liu et al. 2000), which will take any other
    TinyDA proposal as a kernel. If the kernel is symmetric, it uses MTM(II),
    otherwise it uses MTM(I). The parameter k sets the number of tries.

    Attributes
    ----------
    kernel : tinyDA.Proposal
        The kernel of the Multiple-Try proposal (another proposal).
    k : int
        Number of mutiple tries.

    Methods
    ----------
    setup_proposal(**kwargs)
        Initialises the kernel, and the remote Posteriors.
    adapt(**kwargs)
        Adapts the kernel.
    make_proposal(link)
        Generates a Multiple Try proposal, using the kernel.
    get_acceptance(proposal_link, previous_link)
        Computes the acceptance probability given a proposal link and the
        previous link.
    """

    is_symmetric = True

    def __init__(self, kernel, k):
        """
        Parameters
        ----------
        kernel : tinyDA.Proposal
            The kernel of the Multiple-Try proposal (another proposal)
        k : int
            Number of mutiple tries.
        """

        # set the kernel
        self.kernel = kernel

        # set the number of tries per proposal.
        self.k = k

        if self.kernel.adaptive:
            warnings.warn(
                " Using global adaptive scaling with MultipleTry proposal can be unstable.\n"
            )

        ray.init(ignore_reinit_error=True)

    def setup_proposal(self, **kwargs):
        # pass the kwargs to the kernel.
        self.kernel.setup_proposal(**kwargs)

        # initialise the posteriors.
        self.posteriors = [
            RemotePosterior.remote(kwargs["posterior"]) for i in range(self.k)
        ]

    def adapt(self, **kwargs):
        # this method is not adaptive in its own, but its kernel might be.
        self.kernel.adapt(**kwargs)

    def make_proposal(self, link):
        # create proposals. this is fast so no paralellised.
        proposals = [self.kernel.make_proposal(link) for i in range(self.k)]

        # get the links in parallel.
        proposal_processes = [
            posterior.create_link.remote(proposal)
            for proposal, posterior in zip(proposals, self.posteriors)
        ]
        self.proposal_links = ray.get(proposal_processes)

        # if kernel is symmetric, use MTM(II), otherwise use MTM(I).
        if self.kernel.is_symmetric:
            q_x_y = np.zeros(self.k)
        else:
            q_x_y = np.array(
                [
                    self.kernel.get_q(link, proposal_link)
                    for proposal_link in self.proposal_links
                ]
            )

        # get the unnormalised weights.
        self.proposal_weights = np.array(
            [link.posterior + q for link, q in zip(self.proposal_links, q_x_y)]
        )
        self.proposal_weights[np.isnan(self.proposal_weights)] = -np.inf

        # if all posteriors are -Inf, return a random one.
        if np.isinf(self.proposal_weights).all():
            return np.random.choice(self.proposal_links).parameters

        # otherwise, return a random one according to the weights.
        else:
            return np.random.choice(
                self.proposal_links,
                p=np.exp(self.proposal_weights - logsumexp(self.proposal_weights)),
            ).parameters

    def get_acceptance(self, proposal_link, previous_link):
        # check if the proposal makes sense, if not return 0.
        if np.isnan(proposal_link.posterior) or np.isinf(self.proposal_weights).all():
            return 0

        else:
            # create reference proposals.this is fast so no paralellised.
            references = [
                self.kernel.make_proposal(proposal_link) for i in range(self.k - 1)
            ]

            # get the links in parallel.
            reference_processes = [
                posterior.create_link.remote(reference)
                for reference, posterior in zip(references, self.posteriors)
            ]
            self.reference_links = ray.get(reference_processes)

            # if kernel is symmetric, use MTM(II), otherwise use MTM(I).
            if self.kernel.is_symmetric:
                q_y_x = np.zeros(self.k)
            else:
                q_y_x = np.array(
                    [
                        self.kernel.get_q(proposal_link, reference_link)
                        for reference_link in self.reference_links
                    ]
                )

            # get the unnormalised weights.
            self.reference_weights = np.array(
                [link.posterior + q for link, q in zip(self.reference_links, q_y_x)]
            )
            self.reference_weights[np.isnan(self.reference_weights)] = -np.inf

            # get the acceptance probability.
            return np.exp(
                logsumexp(self.proposal_weights) - logsumexp(self.reference_weights)
            )


@ray.remote
class RemotePosterior:
    def __init__(self, posterior):
        self.posterior = posterior

    def create_link(self, parameters):
        return self.posterior.create_link(parameters)

@ray.remote
class ArchiveManager:
    def __init__(self, chain_count, data_len):
        # separate collection for each chain
        self.shared_archive = [None] * chain_count
        self.chain_count = chain_count

    def update_archive(self, sample, chain_id):
        # update the whole collection
        try:
            self.shared_archive[chain_id] = np.vstack((self.shared_archive[chain_id], sample))
        except ValueError:
            self.shared_archive[chain_id] = sample

    def get_archive(self):
        try:
            return np.concatenate(self.shared_archive)
        except ValueError:
            valid_achives = [a for a in self.shared_archive if a is not None]
            return np.concatenate(valid_achives)
