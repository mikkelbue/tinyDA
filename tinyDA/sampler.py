# external imports
import copy
import warnings
from itertools import compress
import scipy.stats as stats

# internal imports
from .chain import *
from .proposal import *
from .distributions import *

# check if ray is available and set a flag.
try:
    from .ray import *

    ray_is_available = True
except ModuleNotFoundError:
    ray_is_available = False


def sample(
    posteriors,
    proposal,
    iterations,
    n_chains=1,
    initial_parameters=None,
    subchain_length=1,
    randomize_subchain_length=False,
    adaptive_error_model=None,
    store_coarse_chain=True,
    force_sequential=False,
    force_progress_bar=False,
    subsampling_rate=None,
):
    """Returns MCMC samples given a tinyDA.Posterior and a tinyDA.Proposal.
    This function takes as input a tinyDA.Posterior instance, or a list of
    tinyDA.Posterior instances. If a single instance is provided, standard
    single-level Metropolis-Hastings MCMC is completed. If a list with two
    tinyDA.Posterior instances is provided, Delayed Acceptance MCMC will be
    completed using the first list item as the coarse model and the second list
    item as the fine model. If a list with more than two instances of
    tda.Posterior is given, MLDA sampling is completed. In this case, the
    list of posteriors must be in increasing order, with the coarsest model
    first and the finest model last. The tinyDA.Posterior instances wrap both
    the prior, likelihood and the forward model(s), and must be initialised
    first. A tinyDA.Proposal and the required number of MCMC iterations must
    also be provided to tinyDA.sample(). The function then returns a dictionary
    containing chain information and MCMC samples in the form of tinyDA.Link
    instances. Use tinyDA.to_inference_data() to convert the output to an
    arviz.InferenceData object.

    Parameters
    ----------
    posteriors : list or tinyDA.Posterior
        A list of tinyDA.Posterior instances or a single instance. If
        posteriors is a single tinyDA.Posterior or a list with one
        tinyDA.Posterior instance, tinyDA.sample() will run standard
        single-level Metropolis-Hastings MCMC. If posteriors is a list
        containing two tinyDA.Posterior instances, tinyDA.sample() will
        run Delayed Acceptance MCMC with the first list item as the
        coarse model and the second list item as the fine model.
    proposal : tinyDA.Proposal
        The MCMC proposal to be used for sampling. If running (Multilevel)
        Delayed Acceptance sampling, the proposal will be used as the
        coarsest proposal.
    iterations : int
        Number of MCMC samples to generate on the finest level.
    n_chains : int, optional
        Number of independent MCMC samplers. Default is 1.
    initial_parameters : list or numpy.ndarray or None, optional
        The parameters of the initial sample. If a list is provided, each
        item will serve as the initial sample of the independent MCMC
        samplers. If a numpy.ndarray is provided, all MCMC sampler will
        have the same initial sample. If None, each MCMC sampler will be
        initialised with a random draw from the prior. Default is None.
    subchain_length : list or int, optional
        The subchain length(s). If subchain_length is a list, it must have
        length len(posteriors) - 1, in increasing order. If it is an int,
        the same subchain length will be used for all levels. If running
        single-level MCMC, this parameter is ignored. Default is 1,
        resulting in "classic" DA MCMC for a two-level model.
    adaptive_error_model : str or None, optional
        The adaptive error model, see e.g. Cui et al. (2019). If running
        single-level MCMC, this parameter is ignored. Default is None
        (no error model), options are 'state-independent' or 'state-dependent'.
        If an error model is used, the likelihood MUST have a set_bias()
        method, use e.g. tinyDA.AdaptiveGaussianLogLike.
    store_coarse_chain : bool, optional
        Whether to store the coarse chain. Disable if the sampler is
        taking up too much memory. Default is True.
    force_sequential : bool, optional
        Whether to force sequential sampling, even if Ray is installed.
        Default is False.
    force_progress_bar : bool, optional
        Whether to force printing of progress bar for parallel sampling.
        This will result in "messy" progress bar output, unless Ray is
        patched.

    Returns
    ----------
    dict
        A dict with keys 'sampler' (which sampler was used, MH or DA),
        'n_chains' (the number of independent MCMC chains),
        'iterations' (the number of MCMC samples in each independent chain),
        'subchain_length' (the Delayed Acceptance subchain length) and
        'chain_1' ... 'chain_n' containing the MCMC samples from each
        independent chain in the form of tinyDA.Link instances. This dict
        can be used as input for tinyDA.to_inference_data() to yield an
        arviz.InferenceData object.
    """


    if subsampling_rate is not None:
        warnings.warn(" subsampling_rate has been deprecated in favour of subchain_length.")
        subchain_length = subsampling_rate

    # get the availability flag.
    global ray_is_available

    # put the posterior in a list, so that it can be indexed.
    if not isinstance(posteriors, list):
        posteriors = [posteriors]

    # set the number of levels.
    n_levels = len(posteriors)

    # do not use MultipleTry proposal with parallel sampling, since that will create nested
    # instances in Ray, which will be competing for resources. This can be very slow.
    if ray_is_available:
        if isinstance(proposal, MultipleTry):
            force_sequential = True
            if n_chains > 1:
                warnings.warn(
                    " MultipleTry proposal is not compatible with parallel sampling. Forcing sequential mode...\n"
                )

    # if the proposal is pCN, make sure that the prior is multivariate normal.
    if isinstance(proposal, CrankNicolson) and not isinstance(
        posteriors[0].prior, stats._multivariate.multivariate_normal_frozen
    ):
        raise TypeError(
            "Prior must be of type scipy.stats.multivariate_normal for pCN proposal"
        )

    # check the same if the CrankNicolson is nested in a MultipleTry proposal.
    elif hasattr(proposal, "kernel"):
        if isinstance(proposal.kernel, CrankNicolson) and not isinstance(
            posteriors[0].prior, stats._multivariate.multivariate_normal_frozen
        ):
            raise TypeError(
                "Prior must be of type scipy.stats.multivariate_normal for pCN kernel"
            )

    # if the proposal is PoissonPointProposal, make sure that the prior is PoissonPointProcess.
    if isinstance(proposal, PoissonPointProposal) and not isinstance(
        posteriors[0].prior, PoissonPointProcess
    ):
        raise TypeError(
            "Prior must be tinyDA.PoissonPointProcess for tinyDA.PoissonPointProposal proposal."
        )

    # similarly, if the prior is PoissonPointProcess, make sure that the proposal is PoissonPointProposal.
    if isinstance(posteriors[0].prior, PoissonPointProcess) and not isinstance(
        proposal, PoissonPointProposal
    ):
        raise TypeError(
            "Proposal must be tinyDA.PoissonPointProposal for tinyDA.PoissonPointProcess prior."
        )

    proposal = [copy.deepcopy(proposal) for i in range(n_chains)]

    # raise a warning if there are more than two levels (not implemented yet).
    if n_levels > 2 and adaptive_error_model == "state-dependent":
        warnings.warn(
            " A state-dedependent adaptive error model for MLDA has not been implemented yet, defaulting to state-independent AEM..."
        )
        adaptive_error_model = "state-independent"

    if adaptive_error_model == "state-dependent" and subchain_length > 1:
        warnings.warn(
            " Using a state-dependent error model for subchain lengths larger than 1 is not guaranteed to be ergodic. \n"
        )

    # check if the given initial parameters are the right type and size.
    if initial_parameters is not None:
        if type(initial_parameters) == list:
            assert (
                len(initial_parameters) == n_chains
            ), "If list of initial parameters is provided, it must have length n_chains"
        elif type(initial_parameters) == np.ndarray:
            assert (
                posteriors[0].prior.rvs().size == initial_parameters.size
            ), "If an array of initial parameters is provided, it must have the same dimension as the prior"
            initial_parameters = [initial_parameters] * n_chains
        else:
            raise TypeError("Initial paramaters must be list, numpy array or None")
    else:
        initial_parameters = [None] * n_chains

    # start the appropriate sampling algorithm.
    # "vanilla" MCMC
    if n_levels == 1:
        # sequential sampling.
        if not ray_is_available or n_chains == 1 or force_sequential:
            samples = _sample_sequential(
                posteriors, proposal, iterations, n_chains, initial_parameters
            )
        # parallel sampling.
        else:
            samples = _sample_parallel(
                posteriors,
                proposal,
                iterations,
                n_chains,
                initial_parameters,
                force_progress_bar,
            )

    # delayed acceptance MCMC.
    elif n_levels == 2:
        # sequential sampling.
        if not ray_is_available or n_chains == 1 or force_sequential:
            samples = _sample_sequential_da(
                posteriors,
                proposal,
                iterations,
                n_chains,
                initial_parameters,
                subchain_length,
                randomize_subchain_length,
                adaptive_error_model,
                store_coarse_chain,
            )
        # parallel sampling.
        else:
            samples = _sample_parallel_da(
                posteriors,
                proposal,
                iterations,
                n_chains,
                initial_parameters,
                subchain_length,
                randomize_subchain_length,
                adaptive_error_model,
                store_coarse_chain,
                force_progress_bar,
            )

    elif n_levels > 2:
        if isinstance(subchain_length, list):
            subchain_lengths = subchain_length
        elif isinstance(subchain_length, int):
            subchain_lengths = [subchain_length] * (len(posteriors) - 1)

        # sequential sampling.
        if not ray_is_available or n_chains == 1 or force_sequential:
            samples = _sample_sequential_mlda(
                posteriors,
                proposal,
                iterations,
                n_chains,
                initial_parameters,
                subchain_lengths,
                adaptive_error_model,
                store_coarse_chain,
            )
        # parallel sampling.
        else:
            samples = _sample_parallel_mlda(
                posteriors,
                proposal,
                iterations,
                n_chains,
                initial_parameters,
                subchain_lengths,
                adaptive_error_model,
                store_coarse_chain,
                force_progress_bar,
            )

    return samples


def _sample_sequential(posteriors, proposal, iterations, n_chains, initial_parameters):
    """Helper function for tinyDA.sample()"""

    # initialise the chains and sample, sequentially.
    chains = []
    for i in range(n_chains):
        print("Sampling chain {}/{}".format(i + 1, n_chains))
        chains.append(Chain(posteriors[0], proposal[i], initial_parameters[i]))
        chains[i].sample(iterations)

    info = {"sampler": "MH", "n_chains": n_chains, "iterations": iterations + 1}
    chains = {"chain_{}".format(i): chain.chain for i, chain in enumerate(chains)}

    # return the samples.
    return {**info, **chains}


def _sample_parallel(
    posteriors,
    proposal,
    iterations,
    n_chains,
    initial_parameters,
    force_progress_bar,
):
    """Helper function for tinyDA.sample()"""

    print("Sampling {} chains in parallel".format(n_chains))

    # create a parallel sampling instance and sample.
    chains = ParallelChain(posteriors[0], proposal, n_chains, initial_parameters)
    chains.sample(iterations, force_progress_bar)

    info = {"sampler": "MH", "n_chains": n_chains, "iterations": iterations + 1}
    chains = {"chain_{}".format(i): chain for i, chain in enumerate(chains.chains)}

    # return the samples.
    return {**info, **chains}


def _sample_sequential_da(
    posteriors,
    proposal,
    iterations,
    n_chains,
    initial_parameters,
    subchain_length,
    randomize_subchain_length,
    adaptive_error_model,
    store_coarse_chain,
):
    """Helper function for tinyDA.sample()"""

    # initialise the chains and sample, sequentially.
    chains = []
    for i in range(n_chains):
        print("Sampling chain {}/{}".format(i + 1, n_chains))
        chains.append(
            DAChain(
                posteriors[0],
                posteriors[1],
                proposal[i],
                subchain_length,
                randomize_subchain_length,
                initial_parameters[i],
                adaptive_error_model,
                store_coarse_chain,
            )
        )
        chains[i].sample(iterations)

    info = {
        "sampler": "DA",
        "n_chains": n_chains,
        "iterations": iterations + 1,
        "subchain_length": subchain_length,
    }

    # collect the coarse samples.
    if store_coarse_chain:
        chains_coarse = {
            "chain_coarse_{}".format(i): list(
                compress(chain.chain_coarse, chain.is_coarse)
            )
            for i, chain in enumerate(chains)
        }
    else:
        chains_coarse = {
            "chain_coarse_{}".format(i): None for i, chain in enumerate(chains)
        }

    # collect the fine samples.
    chains_fine = {
        "chain_fine_{}".format(i): chain.chain_fine for i, chain in enumerate(chains)
    }

    # return eveything.
    return {**info, **chains_coarse, **chains_fine}


def _sample_parallel_da(
    posteriors,
    proposal,
    iterations,
    n_chains,
    initial_parameters,
    subchain_length,
    randomize_subchain_length,
    adaptive_error_model,
    store_coarse_chain,
    force_progress_bar,
):
    """Helper function for tinyDA.sample()"""

    print("Sampling {} chains in parallel".format(n_chains))

    # create a parallel sampling instance and sample.
    chains = ParallelDAChain(
        posteriors[0],
        posteriors[1],
        proposal,
        subchain_length,
        randomize_subchain_length,
        n_chains,
        initial_parameters,
        adaptive_error_model,
        store_coarse_chain,
    )
    chains.sample(iterations, force_progress_bar)

    info = {
        "sampler": "DA",
        "n_chains": n_chains,
        "iterations": iterations + 1,
        "subchain_length": subchain_length,
    }

    # collect the coarse samples.
    chains_coarse = {
        "chain_coarse_{}".format(i): chain[0] for i, chain in enumerate(chains.chains)
    }

    # collect the fine samples.
    chains_fine = {
        "chain_fine_{}".format(i): chain[1] for i, chain in enumerate(chains.chains)
    }

    # return everything.
    return {**info, **chains_coarse, **chains_fine}


def _sample_sequential_mlda(
    posteriors,
    proposal,
    iterations,
    n_chains,
    initial_parameters,
    subchain_lengths,
    adaptive_error_model,
    store_coarse_chain,
):
    """Helper function for tinyDA.sample()"""

    levels = len(posteriors)

    # initialise the chains and sample, sequentially.
    chains = []
    for i in range(n_chains):
        print("Sampling chain {}/{}".format(i + 1, n_chains))
        chains.append(
            MLDAChain(
                posteriors,
                proposal[i],
                subchain_lengths,
                initial_parameters[i],
                adaptive_error_model,
                store_coarse_chain,
            )
        )
        chains[i].sample(iterations)

    info = {
        "sampler": "MLDA",
        "n_chains": n_chains,
        "iterations": iterations + 1,
        "levels": levels,
        "subchain_lengths": subchain_lengths,
    }

    # collect and return the samples.
    chains_all = {
        "chain_l{}_{}".format(levels - 1, i): chain.chain
        for i, chain in enumerate(chains)
    }

    # iterate through the different MLDA levels recursively.
    _current = [chain.proposal for chain in chains]
    for i in reversed(range(levels - 1)):
        if store_coarse_chain:
            chains_current = {
                "chain_l{}_{}".format(i, j): list(compress(chain.chain, chain.is_local))
                for j, chain in enumerate(_current)
            }
        else:
            chains_current = {
                "chain_l{}_{}".format(i, j): None for j, chain in enumerate(_current)
            }
        chains_all = {**chains_all, **chains_current}
        _current = [chain.proposal for chain in _current]

    return {**info, **chains_all}


def _sample_parallel_mlda(
    posteriors,
    proposal,
    iterations,
    n_chains,
    initial_parameters,
    subchain_lengths,
    adaptive_error_model,
    store_coarse_chain,
    force_progress_bar,
):
    """Helper function for tinyDA.sample()"""

    levels = len(posteriors)

    print("Sampling {} chains in parallel".format(n_chains))

    # create a parallel sampling instance and sample.
    chains = ParallelMLDAChain(
        posteriors,
        proposal,
        subchain_lengths,
        n_chains,
        initial_parameters,
        adaptive_error_model,
        store_coarse_chain,
    )
    chains.sample(iterations, force_progress_bar)

    info = {
        "sampler": "MLDA",
        "n_chains": n_chains,
        "iterations": iterations + 1,
        "levels": levels,
        "subchain_lengths": subchain_lengths,
    }

    # iterate through the different chains and MLDA levels recursively.
    chains_all = {}
    for i in range(n_chains):
        for j in range(levels):
            chains_all["chain_l{}_{}".format(j, i)] = chains.chains[i][j]

    return {**info, **chains_all}
