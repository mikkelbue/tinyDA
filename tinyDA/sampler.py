# external imports
import copy
import warnings
from itertools import compress
import scipy.stats as stats

# internal imports
from .chain import *
from .proposal import *

# check if ray is available and set a flag.
try:
    from .ray import *
    ray_is_available = True
except ModuleNotFoundError:
    ray_is_available = False

def sample(link_factory, 
           proposal, 
           iterations, 
           n_chains=1, 
           initial_parameters=None, 
           subsampling_rate=1, 
           adaptive_error_model=None, 
           R=None,
           force_sequential=False,
           force_progress_bar=False):
               
    '''
    Returns MCMC samples given a tinyDA.LinkFactory and a tinyDA.Proposal. 
    This function takes as input a tinyDA.LinkFactory instance, or a list
    of tinyDA.LinkFactory instances. If a single instance is provided,
    standard single-level Metropolis-Hastings MCMC is completed. If a list
    with two tinyDA.LinkFactory instances is provided, Delayed Acceptance
    MCMC will be completed using the first list item as the coarse model
    and the second list item as the fine model. The tinyDA.LinkFactory
    instances wrap both the prior, likelihood and the forward model(s), and 
    must be initialised first. A tinyDA.Proposal and the required number 
    of MCMC iterations must also be provided to tinyDA.sample(). The function 
    then returns a dictionary containing chain information and MCMC samples
    in the form of tinyDA.Link instances. Use tinyDA.to_inference_data() to
    convert the output to an arviz.InferenceData object.
    
    Parameters
    ----------
    link_factory : list or tinyDA.LinkFactory
        A list of tinyDA.LinkFactory instances or a single instance. If
        link_factory is a single tinyDA.LinkFactory or a list with one
        tinyDA.LinkFactory instance, tinyDA.sample() will run standard 
        single-level Metropolis-Hastings MCMC. If link_factory is a list 
        containing two tinyDA.LinkFactory instances, tinyDA.sample() will 
        run Delayed Acceptance MCMC with the first list item as the 
        coarse model and the second list item as the fine model. 
    proposal : tinyDA.Proposal
        The MCMC proposal to be used for sampling. If running Delayed
        Acceptance sampling, the proposal will be used as the coarse 
        proposal.
    iterations : int
        Number of MCMC samples to generate.
    n_chains : int, optional
        Number of independent MCMC samplers. Default is 1.
    initial_parameters : list or numpy.ndarray or None, optional
        The parameters of the initial sample. If a list is provided, each
        item will serve as the initial sample of the independent MCMC 
        samplers. If a numpy.ndarray is provided, all MCMC sampler will
        have the same initial sample. If None, each MCMC sampler will be
        initialised with a random draw from the prior. Default is None.
    subsampling_rate : int, optional
        The subsampling rate or subchain length used for Delayed Acceptance
        sampling. If running single-level MCMC, this parameter is ignored.
        Default is 1, resulting in "classic" DA MCMC.
    adaptive_error_model : str or None, optional
        The adaptive error model, see e.g. Cui et al. (2019). If running 
        single-level MCMC, this parameter is ignored. Default is None 
        (no error model), options are 'state-independent' or 'state-dependent'. 
        If an error model is used, the likelihood MUST have a set_bias() 
        method, use e.g. tinyDA.AdaptiveLogLike.
    R : numpy.ndarray or None, optional
        Restriction matrix for the adaptive error model. If running 
        single-level MCMC, this parameter is ignored. Default is None 
        (identity matrix).
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
        'subsampling_rate' (the Delayed Acceptance subsampling rate) and
        'chain_1' ... 'chain_n' containing the MCMC samples from each 
        independent chain in the form of tinyDA.Link instances. This dict
        can be used as input for tinyDA.to_inference_data() to yield an 
        arviz.InferenceData object.
    '''

    # get the availability flag.
    global ray_is_available
    
    # put the link factory in a list, so that it can be indexed.
    if not isinstance(link_factory, list):
        link_factory = [link_factory]
        
    # set the number of levels.
    n_levels = len(link_factory)
    
    # raise a warning if there are more than two levels (not implemented yet).
    if n_levels > 2:
        raise NotImplementedError('MLDA sampling has not been implemented yet. Please input list of 1 or 2 LinkFactory instances')
        
    # do not use MultipleTry proposal with parallel sampling, since that will create nested 
    # instances in Ray, which will be competing for resources. This can be very slow.
    if ray_is_available:
        if isinstance(proposal, MultipleTry):
            force_sequential = True
            if n_chains > 1:
                warnings.warn(' MultipleTry proposal is not compatible with parallel sampling. Forcing sequential mode...\n')

    # if the proposal is pCN, make sure that the prior is multivariate normal.
    if isinstance(proposal, CrankNicolson) and not isinstance(link_factory[0].prior, stats._multivariate.multivariate_normal_frozen):
        raise TypeError('Prior must be of type scipy.stats.multivariate_normal for pCN proposal')
    
    # check the same if the CrankNicolson is nested in a MultipleTry proposal.
    elif hasattr(proposal, 'kernel'):
        if isinstance(proposal.kernel, CrankNicolson) and not isinstance(link_factory[0].prior, stats._multivariate.multivariate_normal_frozen):
            raise TypeError('Prior must be of type scipy.stats.multivariate_normal for pCN kernel')
            
    proposal = [copy.deepcopy(proposal) for i in range(n_chains)]
            
    if adaptive_error_model == 'state-dependent' and subsampling_rate > 1:
        warnings.warn(' Using a state-dependent error model for subsampling rates larger than 1 is not guaranteed to be ergodic. \n')
            
    # check if the given initial parameters are the right type and size.
    if initial_parameters is not None:
        if type(initial_parameters) == list:
            assert len(initial_parameters) == n_chains, 'If list of initial parameters is provided, it must have length n_chains'
        elif type(initial_parameters) == np.ndarray:
            assert link_factory[0].prior.rvs().size == initial_parameters.size, 'If an array of initial parameters is provided, it must have the same dimension as the prior'
            initial_parameters = [initial_parameters]*n_chains
        else:
            raise TypeError('Initial paramaters must be list, numpy array or None')
    else:
        initial_parameters = [None]*n_chains
            
    # start the appropriate sampling algorithm.
    # "vanilla" MCMC
    if n_levels == 1:
        # sequential sampling.
        if not ray_is_available or n_chains == 1 or force_sequential:
            samples = _sample_sequential(link_factory, proposal, iterations, n_chains, initial_parameters)
        # parallel sampling.
        else:
            samples = _sample_parallel(link_factory, proposal, iterations, n_chains, initial_parameters, force_progress_bar)
    
    # delayed acceptance MCMC.
    elif n_levels == 2:
        # sequential sampling.
        if not ray_is_available or n_chains == 1 or force_sequential:
            samples = _sample_sequential_da(link_factory, proposal, iterations, n_chains, initial_parameters, subsampling_rate, adaptive_error_model, R)
        # parallel sampling.
        else:
            samples = _sample_parallel_da(link_factory, proposal, iterations, n_chains, initial_parameters, subsampling_rate, adaptive_error_model, R, force_progress_bar)
    
    return samples


def _sample_sequential(link_factory, proposal, iterations, n_chains, initial_parameters):
    '''
    Helper function for tinyDA.sample()
    '''
    
    # initialise the chains and sample, sequentially.
    chains = []
    for i in range(n_chains):
        print('Sampling chain {}/{}'.format(i+1, n_chains))
        chains.append(Chain(link_factory[0], proposal[i], initial_parameters[i]))
        chains[i].sample(iterations)
    
    info = {'sampler': 'MH', 'n_chains': n_chains, 'iterations': iterations+1}
    chains = {'chain_{}'.format(i): chain.chain for i, chain in enumerate(chains)}
    
    # return the samples.
    return {**info, **chains}
    
def _sample_parallel(link_factory, proposal, iterations, n_chains, initial_parameters, force_progress_bar):
    '''
    Helper function for tinyDA.sample()
    '''
    
    print('Sampling {} chains in parallel'.format(n_chains))
    
    # create a parallel sampling instance and sample.
    chains = ParallelChain(link_factory[0], proposal, n_chains, initial_parameters)
    chains.sample(iterations, force_progress_bar)
    
    info = {'sampler': 'MH', 'n_chains': n_chains, 'iterations': iterations+1}
    chains = {'chain_{}'.format(i): chain for i, chain in enumerate(chains.chains)}
    
    # return the samples.
    return {**info, **chains}
    
def _sample_sequential_da(link_factory, proposal, iterations, n_chains, initial_parameters, subsampling_rate, adaptive_error_model, R):
    '''
    Helper function for tinyDA.sample()
    '''
    
    # initialise the chains and sample, sequentially.
    chains = []
    for i in range(n_chains):
        print('Sampling chain {}/{}'.format(i+1, n_chains))
        chains.append(DAChain(link_factory[0], link_factory[1], proposal[i], subsampling_rate, initial_parameters[i], adaptive_error_model, R))
        chains[i].sample(iterations)
    
    info = {'sampler': 'DA', 'n_chains': n_chains, 'iterations': iterations+1, 'subsampling_rate': subsampling_rate}
    
    # collect and return the samples.
    chains_coarse = {'chain_coarse_{}'.format(i): list(compress(chain.chain_coarse, chain.is_coarse)) for i, chain in enumerate(chains)}
    chains_fine = {'chain_fine_{}'.format(i): chain.chain_fine for i, chain in enumerate(chains)}
        
    return {**info, **chains_coarse, **chains_fine}

def _sample_parallel_da(link_factory, proposal, iterations, n_chains, initial_parameters, subsampling_rate, adaptive_error_model, R, force_progress_bar):
    '''
    Helper function for tinyDA.sample()
    '''
    
    print('Sampling {} chains in parallel'.format(n_chains))
    
    # create a parallel sampling instance and sample.
    chains = ParallelDAChain(link_factory[0], link_factory[1], proposal, subsampling_rate, n_chains, initial_parameters, adaptive_error_model, R)
    chains.sample(iterations, force_progress_bar)
    
    info = {'sampler': 'DA', 'n_chains': n_chains, 'iterations': iterations+1, 'subsampling_rate': subsampling_rate}
    
    # collect and return the samples.
    chains_coarse = {'chain_coarse_{}'.format(i): chain[0] for i, chain in enumerate(chains.chains)}
    chains_fine = {'chain_fine_{}'.format(i): chain[1] for i, chain in enumerate(chains.chains)}
        
    return {**info, **chains_coarse, **chains_fine}
