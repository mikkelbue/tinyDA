from .chain import *

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

    # get the availability flag.
    global ray_is_available
    
    # put the link factory in a list, so that it can be indexed.
    if not isinstance(link_factory, list):
        link_factory = [link_factory]
        
    # set the number of levels.
    n_levels = len(link_factory)
    
    # raise a warning if there are more than two levels (not implemented yet).
    if n_levels > 2:
        raise NotImplementedError('MLDA sampling has not been implemented yet. Please input 1 or 2 LinkFactory instances')
        
    # do not use MultipleTry proposal with parallel sampling, since that will create nested 
    # instances in Ray, which will be competing for resources. This can be very slow.
    if isinstance(proposal, MultipleTry):
        force_sequential = True

    # if the proposal is pCN, make sure that the prior is multivariate normal.
    if isinstance(proposal, CrankNicolson) and not isinstance(link_factory[0].prior, stats._multivariate.multivariate_normal_frozen):
        raise TypeError('Prior must be of type scipy.stats.multivariate_normal for pCN proposal')
    
    # check the same if the CrankNicolson is nested in a MultipleTry proposal.
    elif hasattr(proposal, 'kernel'):
        if isinstance(proposal.kernel, CrankNicolson) and not isinstance(link_factory[0].prior, stats._multivariate.multivariate_normal_frozen):
            raise TypeError('Prior must be of type scipy.stats.multivariate_normal for pCN kernel')
            
    # check if the given initial parameters are the right type and size.
    if initial_parameters is not None:
        if type(initial_parameters) == list:
            assert len(initial_parameters) == n_chains, 'If list of initial parameters is provided, it must have length n_chains'
        elif type(initial_parameters) == np.ndarray:
            assert link_factory[0].prior.rvs().size == initial_parameters.size, 'If an array of initial parameters is provided, it must have the same dimension as the prior'
            initial_parameters = [initial_parameters]*n_chains
        else:
            raise TypeError('Initial paramaters must be list, numpy array or None')
            
    # start the appropriate sampling algorithm.
    # "vanilla" MCMC
    if n_levels == 1:
        # sequential sampling.
        if not ray_is_available or chains==1 or force_sequential:
            samples = _sample_sequential(link_factory, proposal, iterations, n_chains, initial_parameters)
        # parallel sampling.
        else:
            samples = _sample_parallel(link_factory, proposal, iterations, n_chains, initial_parameters, force_progress_bar)
    
    # delayed acceptance MCMC.
    elif n_levels == 2:
        # sequential sampling.
        if not ray_is_available or chains=1 or force_sequential:
            samples = _sample_sequential_da(link_factory, proposal, iterations, n_chains, initial_parameters, subsampling_rate, adaptive_error_model, R)
        # parallel sampling.
        else:
            samples = _sample_parallel_da(link_factory, proposal, iterations, n_chains, initial_parameters, subsampling_rate, adaptive_error_model, R, force_progress_bar)
    
    return samples


def _sample_sequential(link_factory, proposal, iterations, n_chains, initial_parameters):
    
    # create a list of None to allow iteration.
    if initial_parameters is None:
        initial_parameters = [None]*n_chains
    
    # initialise the chains and sample, sequentially.
    chains = []
    for i in range(n_chains):
        print('Sampling chain {}/{}'.format(i, n_chains))
        chains.append(Chain(link_factory, proposal, initial_parameters[i]))
        chains[i].sample(iterations)

    # return the samples.
    return {'chain_{}'.format(i): chain.chain for i, chain in enumerate(chains)}
    
def _sample_parallel(link_factory, proposal, iterations, n_chains, initial_parameters, force_progress_bar):
    
    print('Sampling {} chains in parallel'.format(n_chains))
    
    # create a parallel sampling instance and sample.
    chains = ParallelChain(link_factory, proposal, n_chains, initial_parameters)
    chains.sample(iterations, force_progress_bar)
    
    # return the samples.
    return {'chain_{}'.format(i): chain for i, chain in enumerate(chains.chains)}
    
def _sample_sequential_da(link_factory, proposal, iterations, n_chains, initial_parameters, subsampling_rate, adaptive_error_model, R):
    
    # create a list of None to allow iteration.
    if initial_parameters is None:
        initial_parameters = [None]*n_chains
    
    # initialise the chains and sample, sequentially.
    chains = []
    for i in range(n_chains):
        print('Sampling chain {}/{}'.format(i, n_chains))
        chains.append(DAChain(link_factory[0], link_factory[1], proposal, subsampling_rate, initial_parameters[i], adaptive_error_model, R))
        chains[i].sample(iterations, )
    
    # collect and return the samples.
    chains_coarse = {'chain_coarse_{}'.format(i): chain.chain_coarse for i, chain in enumerate(chains)}
    chains_fine = {'chain_fine_{}'.format(i): chain.chain_fine for i, chain in enumerate(chains)}
        
    return {**chains_coarse, **chains_fine}

def _sample_parallel_da(link_factory, proposal, iterations, n_chains, initial_parameters, subsampling_rate, adaptive_error_model, R, force_progress_bar):
    
    print('Sampling {} chains in parallel'.format(n_chains))
    
    # create a parallel sampling instance and sample.
    chains = ParallelDAChain(link_factory_coarse, link_factory_fine, proposal, subsampling_rate, n_chains, initial_parameters, adaptive_error_model, R)
    chains.sample(iterations, force_progress_bar)
    
    # collect and return the samples.
    chains_coarse = {'chain_coarse_{}'.format(i): chain[0] for i, chain in enumerate(chains.chains)}
    chains_fine = {'chain_fine_{}'.format(i): chain[1] for i, chain in enumerate(chains.chains)}
        
    return {**chains_coarse, **chains_fine}
