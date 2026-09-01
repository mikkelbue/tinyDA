import numpy as np
import xarray as xr
import arviz as az


def to_inference_data(chain, level="fine", burnin=0, parameter_names=None):
    """Converts a dict of tinyDA.Link samples as returned by tinyDA.sample() to
    an arviz.InferenceData object. This can be used after running
    tinyDA.sample() to make use of the diagnostics suite provided by ArviZ for
    postprocessing.

    Parameters
    ----------
    chain : dict
        A dict of MCMC samples, as returned by tinyDA.sample().
    level : str, optional
        Which level to extract samples from ('fine', 'coarse').
        If input is single-level MCMC, this parameter is ignored.
        The default is 'fine'.
    burnin : int, optional
        The burnin length. The default is 0.
    parameter_names : list, optional
        List of the names of the parameters in the chain, in the same order 
        as they appear in each link. Default is None, meaning that
        parameters will be named [x1, x2, ...].

    Returns
    ----------
    arviz.InferenceData
        An arviz.InferenceData object containing xarray.Dataset instances
        representative of the MCMC samples.
    """

    # set the attributes that will be included in the InferenceData instance.
    attributes = ["parameters", "model_output", "qoi", "stats"]

    # initialise a list to hold the xarray.Dataset instances.
    inference_arrays = []

    # iterate through the attributes and create xarray.Datasets
    for attr in attributes:

        samples = get_samples(chain, attr, level, burnin)

        # set up the dict keys to reflect the extracted attribute.
        if attr == "parameters":
            if parameter_names is None:
                keys = ["x{}".format(i) for i in range(samples["dimension"])]
            else:
                keys = parameter_names
        elif attr == "model_output":
            keys = ["obs_{}".format(i) for i in range(samples["dimension"])]
        elif attr == "qoi":
            keys = ["qoi_{}".format(i) for i in range(samples["dimension"])]
        elif attr == "stats":
            keys = ["prior", "likelihood", "posterior"]

        inference_arrays.append(to_xarray(samples, keys))

    # create the InferenceData instance.
    idata = az.InferenceData(
        posterior=inference_arrays[0],
        posterior_predictive=inference_arrays[1],
        qoi=inference_arrays[2],
        sample_stats=inference_arrays[3],
    )

    # return InferenceData,
    return idata


def to_xarray(samples, keys):
    """Converts a dict of attribute samples to an xarray.Dataset.

    Parameters
    ----------
    samples : dict
        A dict of MCMC samples, as returned by tinyDA.get_samples().
    keys : list
        Names of the variables of the attribute.

    Returns
    ----------
    xarray.Dataset
        An xarray.Dataset with coordinates 'chain' and 'draw', corresponding
        to independent MCMC sampler and their respective samples.
    """

    # initialise a dict to hold the data variables.
    data_vars = {}  #

    # iterate through the data variables.
    for i in range(samples["dimension"]):
        # extract and pivot the data variables.
        x = np.array(
            [samples["chain_{}".format(j)][:, i] for j in range(samples["n_chains"])]
        )
        # add the coordinates to the data variables.
        data_vars[keys[i]] = (["chain", "draw"], x)

    # create the dataset.
    dataset = xr.Dataset(
        data_vars=data_vars,
        coords=dict(
            chain=("chain", list(range(samples["n_chains"]))),
            draw=("draw", list(range(samples["iterations"]))),
        ),
    )

    # return the dataset.
    return dataset


def get_samples(chain, attribute="parameters", level="fine", burnin=0):
    """Converts a dict of tinyDA.Link samples as returned by tinyDA.sample() to
    a dict of numpy.ndarrays corresponding to the MCMC samples of the required
    tinyDA.Link attribute. Possible attributes are 'parameters', which returns
    the parameters of each sample, 'model_output', which returns the model
    response F(theta) for each sample, 'qoi', which returns the quantity of
    interest for each sample and 'stats', which returns the log-prior, log-
    likelihood and log-posterior of each sample.

    Parameters
    ----------
    chain : dict
        A dict as returned by tinyDA.sample, containing chain information
        and lists of tinyDA.Link instances.
    attribute : str, optional
        Which link attribute ('parameters', 'model_output', 'qoi' or 'stats')
        to extract. The default is 'parameters'.
    level : str, optional
        Which level to extract samples from ('fine', 'coarse').
        If input is single-level MCMC, this parameter is ignored.
        The default is 'fine'.
    burnin : int, optional
        The burnin length. The default is 0.

    Returns
    ----------
    dict
        A dict of numpy array(s) with the parameters or the qoi as columns
        and samples as rows.
    """

    # copy some items across.
    samples = {
        "sampler": chain["sampler"],
        "n_chains": chain["n_chains"],
        "attribute": attribute,
    }

    if attribute == 'stats':
        getattribute = lambda link, attribute: np.array([link.prior, link.likelihood, link.posterior])
    else:
        getattribute = lambda link, attribute: getattr(link, attribute)

    # if the input is a single-level Metropolis-Hastings chain.
    if chain["sampler"] == "MH":
        # extract link attribute.
        for i in range(chain["n_chains"]):
            samples["chain_{}".format(i)] = np.array(
                [getattribute(link, attribute) for link in chain["chain_{}".format(i)][burnin:]]
            )

    # if the input is a Delayed Acceptance chain.
    elif chain["sampler"] == "DA":
        # copy the subchain length across.
        samples["subchain_length"] = chain["subchain_length"]
        # set the extraction level ('coarse' or 'fine').
        samples["level"] = level
        # extract attribute
        for i in range(chain["n_chains"]):
            samples["chain_{}".format(i)] = np.array(
                [
                    getattribute(link, attribute)
                    for link in chain["chain_{}_{}".format(level, i)][burnin:]
                ]
            )


        # if the input is a Delayed Acceptance chain.
    elif chain["sampler"] == "MLDA":
        # copy the subchain length across.
        samples["subchain_lengths"] = chain["subchain_lengths"]
        # set the extraction level.
        samples["level"] = level
        # extract attribute
        for i in range(chain["n_chains"]):
            samples["chain_{}".format(i)] = np.array(
                [
                    getattribute(link, attribute)
                    for link in chain["chain_l{}_{}".format(level, i)][burnin:]
                ]
            )

    # expand the dimension of the output, if the required attribute is one-dimensional.
    for i in range(chain["n_chains"]):
        if samples["chain_{}".format(i)].ndim == 1:
            samples["chain_{}".format(i)] = samples["chain_{}".format(i)][
                ..., np.newaxis
            ]

    # add the iterations after subtracting burnin to the output dict.
    samples["iterations"] = samples["chain_0"].shape[0]
    # add the dimension of the attribute to the output dict.
    samples["dimension"] = samples["chain_0"].shape[1]

    # return the samples.
    return samples

def get_promoted_samples(chain, attribute="parameters", level=0, burnin=0):
    """Extracts promoted samples from MLDA chains at a specified level.
    
    Parameters
    ----------
    chain : dict
        A dict as returned by tinyDA.sample, containing chain information
        and lists of tinyDA.Link instances.
    attribute : str, optional
        Which link attribute ('parameters', 'model_output', 'qoi' or 'stats')
        to extract. The default is 'parameters'.
    level : int, optional
        Which level to extract promoted samples from. The default is 0.
        Cannot be the top level (finest level).
    burnin : int, optional
        The burnin length for the next higher level. The default is 0.

    Returns
    ----------
    dict
        A dict of numpy array(s) with the promoted samples, following the
        same structure as get_samples().
    """
    
    # Check if this is an MLDA chain
    if chain["sampler"] != "MLDA":
        raise ValueError("Function only works with MLDA chains.")
    
    # Check if level is valid
    if level >= chain["levels"] - 1:
        raise ValueError(f"Level {level} cannot promote samples. Top level is {chain['levels'] - 1}.")
    
    # Calculate burnin for promoted samples (use next higher level's burnin)
    subchain_lengths = chain["subchain_lengths"]
    cumulative_factor = 1
    for upper_level in range(level + 1, chain["levels"] - 1):
        cumulative_factor *= subchain_lengths[upper_level]
    promoted_burnin = burnin * cumulative_factor
    
    # Copy some items across
    promoted = {
        "sampler": chain["sampler"],
        "n_chains": chain["n_chains"],
        "level": level,
        "attribute": attribute,
        "burnin": promoted_burnin,
    }
    
    # Set up the getattribute function
    if attribute == 'stats':
        getattribute = lambda link, attribute: np.array([link.prior, link.likelihood, link.posterior])
    else:
        getattribute = lambda link, attribute: getattr(link, attribute)
    
    # Extract promoted samples for each chain
    for i in range(chain["n_chains"]):
        promoted_key = f"promoted_l{level}_{i}"
        
        if promoted_key in chain and chain[promoted_key] is not None:
            promoted_samples = chain[promoted_key][promoted_burnin:]
            
            if len(promoted_samples) > 0:
                promoted[f"chain_{i}"] = np.array(
                    [getattribute(link, attribute) for link in promoted_samples]
                )
            else:
                promoted[f"chain_{i}"] = np.array([])
        else:
            # If no promoted samples exist, return empty array
            promoted[f"chain_{i}"] = np.array([])
    
    # Handle case where promoted samples exist
    if promoted[f"chain_0"].size > 0:
        # Expand dimension if needed (same as get_samples)
        for i in range(chain["n_chains"]):
            if promoted[f"chain_{i}"].ndim == 1:
                promoted[f"chain_{i}"] = promoted[f"chain_{i}"][..., np.newaxis]
        
        # Add iterations and dimension info
        promoted["iterations"] = promoted["chain_0"].shape[0]
        promoted["dimension"] = promoted["chain_0"].shape[1]
    else:
        promoted["iterations"] = 0
        promoted["dimension"] = 0
    
    return promoted

def get_multilevel_inference_data(chain, attribute="parameters", parameter_names=None, burnin=0):
    """Extracts all chains and promoted chains from MLDA sampling.
    
    Parameters
    ----------
    chain : dict
        A dict as returned by tinyDA.sample, containing MLDA chain information
        and lists of tinyDA.Link instances.
    attribute : str, optional
        Which link attribute ('parameters', 'model_output', 'qoi' or 'stats')
        to extract. The default is 'parameters'.
    parameter_names : list, optional
        List of parameter names. Default is None.
    burnin : int, optional
        The burnin length for the top level. Lower levels use scaled burnin
        based on subchain lengths. The default is 0.

    Returns
    ----------
    dict
        A dict containing 'chains' and 'promoted' with all level/chain data.
    """
    
    if chain["sampler"] != "MLDA":
        raise ValueError("Function only works with MLDA chains.")
    
    levels = chain["levels"]
    n_chains = chain["n_chains"]
    subchain_lengths = chain["subchain_lengths"]
    
    # Calculate burnin for each level
    level_burnins = [0] * levels
    level_burnins[levels - 1] = burnin  # Top level
    
    # For lower levels, multiply by cumulative subchain lengths
    for level in range(levels - 2, -1, -1):
        cumulative_factor = 1
        for upper_level in range(level, levels - 1):
            cumulative_factor *= subchain_lengths[upper_level]
        level_burnins[level] = burnin * cumulative_factor
    
    # Set up attribute handling
    if attribute == 'stats':
        getattribute = lambda link, attribute: np.array([link.prior, link.likelihood, link.posterior])
    else:
        getattribute = lambda link, attribute: getattr(link, attribute)
    
    multilevel_inference_data = {
        "sampler": chain["sampler"],
        "levels": levels,
        "n_chains": n_chains,
        "attribute": attribute,
        "burnin": burnin,
        "level_burnins": level_burnins,
        "chains": {},
        "promoted": {}
    }
    
    # Loop through all levels
    for level in range(levels):
        # Extract chains for this level
        for chain_idx in range(n_chains):
            chain_key = f"chain_l{level}_{chain_idx}"
            
            if chain_key in chain and chain[chain_key] is not None:
                chain_samples = chain[chain_key]
                
                # Apply burnin for this level
                chain_burnin = level_burnins[level]
                
                # For top level, remove first element (initial sample) plus burnin
                if level == levels - 1:
                    chain_samples = chain_samples[1 + chain_burnin:]
                else:
                    chain_samples = chain_samples[chain_burnin:]
                
                if len(chain_samples) > 0:
                    samples = np.array([getattribute(link, attribute) for link in chain_samples])
                    
                    # Expand dimension if needed
                    if samples.ndim == 1:
                        samples = samples[..., np.newaxis]
                    
                    multilevel_inference_data["chains"][f"level{level}_chain{chain_idx}"] = samples
        
        # Extract promoted samples (not for top level)
        if level < levels - 1:
            for chain_idx in range(n_chains):
                promoted_key = f"promoted_l{level}_{chain_idx}"
                
                if promoted_key in chain and chain[promoted_key] is not None:
                    promoted_samples = chain[promoted_key]
                    
                    # Apply burnin for promoted samples - use next higher level's burnin
                    promoted_burnin = level_burnins[level + 1]
                    promoted_samples = promoted_samples[promoted_burnin:]
                    
                    if len(promoted_samples) > 0:
                        samples = np.array([getattribute(link, attribute) for link in promoted_samples])
                        
                        # Expand dimension if needed
                        if samples.ndim == 1:
                            samples = samples[..., np.newaxis]
                        
                        multilevel_inference_data["promoted"][f"level{level}_chain{chain_idx}"] = samples
    
    # Add dimension info if chains exist
    if multilevel_inference_data["chains"]:
        first_chain = list(multilevel_inference_data["chains"].values())[0]
        multilevel_inference_data["dimension"] = first_chain.shape[1]
    else:
        multilevel_inference_data["dimension"] = 0
    
    return multilevel_inference_data