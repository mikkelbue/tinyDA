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

# work in progress

def get_DA_samples(chain): 
    # creates dict with additional keyword "fine_correction" 
    return 0

def DA_estimator(DA_samples):
    # computes the MLDA estimator of some qoi w.r.t the posterior
    return 0