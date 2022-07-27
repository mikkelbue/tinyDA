import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import pickle
from itertools import compress

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import xarray as xr
import arviz as az

from scipy.stats import norm, rankdata
from scipy.ndimage import convolve

def to_inference_data(chain, level='fine', burnin=0):
    '''
    Converts a dict of tinyDA.Link samples as returned by tinyDA.sample() 
    to an arviz.InferenceData object. This can be used after running
    tinyDA.sample() to make use of the diagnostics suite provided by
    ArviZ for postprocessing.
    
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
        
    Returns
    ----------
    arviz.InferenceData
        An arviz.InferenceData object containing xarray.Dataset instances
        representative of the MCMC samples.
    '''
    
    # set the attributes that will be included in the InferenceData instance.
    attributes = ['parameters', 'model_output', 'qoi', 'stats']
    
    # initialise a list to hold the xarray.Dataset instances.
    inference_arrays = []
    
    # iterate through the attributes and create xarray.Datasets
    for attr in attributes:
        inference_arrays.append(to_xarray(get_samples(chain, attr, level, burnin)))
    
    # create the InferenceData instance.
    idata = az.InferenceData(posterior=inference_arrays[0],
                             posterior_predictive=inference_arrays[1],
                             qoi=inference_arrays[2],
                             sample_stats=inference_arrays[3])
    
    # return InferenceData,
    return idata

def to_xarray(samples):
    '''
    Converts a dict of attribute samples to an xarray.Dataset.
    
    Parameters
    ----------
    samples : dict
        A dict of MCMC samples, as returned by tinyDA.get_samples().
        
    Returns
    ----------
    xarray.Dataset
        An xarray.Dataset with coordinates 'chain' and 'draw', corresponding
        to independent MCMC sampler and their respective samples.
    '''
    
    # set up the dict keys to reflect the extracted attribute.
    if samples['attribute'] == 'parameters':
        keys = ['theta_{}'.format(i) for i in range(samples['dimension'])]
    elif samples['attribute'] == 'model_output':
        keys = ['obs_{}'.format(i) for i in range(samples['dimension'])]
    elif samples['attribute'] == 'qoi':
        keys = ['qoi_{}'.format(i) for i in range(samples['dimension'])]
    elif samples['attribute'] == 'stats':
        keys = ['prior', 'likelihood', 'posterior']
    
    # initialise a dict to hold the data variables.
    data_vars = {}#
    
    # iterate through the data variables.
    for i in range(samples['dimension']):
        # extract and pivot the data variables.
        theta = np.array([samples['chain_{}'.format(j)][:,i] for j in range(samples['n_chains'])])
        # add the coordinates to the data variables.
        data_vars[keys[i]] = (['chain', 'draw'], theta)
    
    # create the dataset.
    dataset = xr.Dataset(data_vars=data_vars,
                         coords=dict(chain=('chain', list(range(samples['n_chains']))),
                                     draw=('draw', list(range(samples['iterations'])))))
    
    # return the dataset.
    return dataset

def get_samples(chain, attribute='parameters', level='fine', burnin=0):
    '''
    Converts a dict of tinyDA.Link samples as returned by tinyDA.sample() 
    to a dict of numpy.ndarrays corresponding to the MCMC samples of the 
    required tinyDA.Link attribute. Possible attributes are 'parameters',
    which returns the parameters of each sample, 'model_output', which 
    returns the model response F(theta) for each sample, 'qoi', which
    returns the quantity of interest for each sample and 'stats', which
    returns the log-prior, log-likelihood and log-posterior of each sample.
    
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
    '''
    
    # copy some items across.
    samples = {'sampler': chain['sampler'],
               'n_chains': chain['n_chains'], 
               'attribute': attribute}
    
    # if the input is a single-level Metropolis-Hastings chain.
    if chain['sampler'] == 'MH':
        # extract link parameters.
        if attribute == 'parameters':
            for i in range(chain['n_chains']):
                samples['chain_{}'.format(i)] = np.array([link.parameters for link in chain['chain_{}'.format(i)][burnin:]])
        # extract the model output.
        elif attribute == 'model_output':
            for i in range(chain['n_chains']):
                samples['chain_{}'.format(i)] = np.array([link.model_output for link in chain['chain_{}'.format(i)][burnin:]])
        # extract the quantity of interest.
        elif attribute == 'qoi':
            for i in range(chain['n_chains']):
                samples['chain_{}'.format(i)] = np.array([link.qoi for link in chain['chain_{}'.format(i)][burnin:]])
        # extract the stats, i.e. log-prior, log-likelihood and log-posterior.
        elif attribute == 'stats':
            for i in range(chain['n_chains']):
                samples['chain_{}'.format(i)] = np.array([np.array([link.prior, link.likelihood, link.posterior]) for link in chain['chain_{}'.format(i)][burnin:]])
    
    # if the input is a Delayed Acceptance chain.
    elif chain['sampler'] == 'DA':
        # copy the subsampling rate across.
        samples['subsampling_rate'] = chain['subsampling_rate']
        # set the extraction level ('coarse' or 'fine').
        samples['level'] = level
        # extract link parameters.
        if attribute == 'parameters':
            for i in range(chain['n_chains']):
                samples['chain_{}'.format(i)] = np.array([link.parameters for link in chain['chain_{}_{}'.format(level, i)][burnin:]])
        # extract the model output.
        elif attribute == 'model_output':
            for i in range(chain['n_chains']):
                samples['chain_{}'.format(i)] = np.array([link.model_output for link in chain['chain_{}_{}'.format(level, i)][burnin:]])
        # extract the quantity of interest.
        elif attribute == 'qoi':
            for i in range(chain['n_chains']):
                samples['chain_{}'.format(i)] = np.array([link.qoi for link in chain['chain_{}_{}'.format(level, i)][burnin:]])
        # extract the stats, i.e. log-prior, log-likelihood and log-posterior.
        elif attribute == 'stats':
            for i in range(chain['n_chains']):
                samples['chain_{}'.format(i)] = np.array([np.array([link.prior, link.likelihood, link.posterior]) for link in chain['chain_{}_{}'.format(level, i)][burnin:]])
    
        # if the input is a Delayed Acceptance chain.
    elif chain['sampler'] == 'MLDA':
        # copy the subsampling rate across.
        samples['subsampling_rates'] = chain['subsampling_rates']
        # set the extraction level.
        samples['level'] = level
        # extract link parameters.
        if attribute == 'parameters':
            for i in range(chain['n_chains']):
                samples['chain_{}'.format(i)] = np.array([link.parameters for link in chain['chain_l{}_{}'.format(level, i)][burnin:]])
        # extract the model output.
        elif attribute == 'model_output':
            for i in range(chain['n_chains']):
                samples['chain_{}'.format(i)] = np.array([link.model_output for link in chain['chain_l{}_{}'.format(level, i)][burnin:]])
        # extract the quantity of interest.
        elif attribute == 'qoi':
            for i in range(chain['n_chains']):
                samples['chain_{}'.format(i)] = np.array([link.qoi for link in chain['chain_l{}_{}'.format(level, i)][burnin:]])
        # extract the stats, i.e. log-prior, log-likelihood and log-posterior.
        elif attribute == 'stats':
            for i in range(chain['n_chains']):
                samples['chain_{}'.format(i)] = np.array([np.array([link.prior, link.likelihood, link.posterior]) for link in chain['chain_l{}_{}'.format(level, i)][burnin:]])

    # expand the dimension of the output, if the required attribute is one-dimensional.
    for i in range(chain['n_chains']):
        if samples['chain_{}'.format(i)].ndim == 1:
            samples['chain_{}'.format(i)] = samples['chain_{}'.format(i)][..., np.newaxis]
    
    # add the iterations after subtracting burnin to the output dict.
    samples['iterations'] = samples['chain_0'].shape[0]
    # add the dimension of the attribute to the output dict.
    samples['dimension'] = samples['chain_0'].shape[1]
    
    # return the samples.
    return samples
    

def plot_samples(samples, indices=[0, 1], plot_type='trace'):
    '''
    Plot either traceplots or histograms of MCMC samples. The input must
    be a nxd array or list of nxd arrays, where n is the number of samples 
    and d is the parameter dimension. Legacy function.
    
    Parameters
    ----------
    samples : dict
        A dict with numpy array(s) containing samples.
    indices : list, optional
        Which indices to plot. The default is [0,1].
    plot_type : str, optional
        The plot type. Can be 'trace or 'histogram'. The default is 'trace'.
    '''

    warnings.warn(' plot_samples() has been deprecated. Please use to_inference_data() and ArviZ for posterior analysis.')

    samples = np.vstack([samples['chain_{}'.format(i)] for i in range(samples['n_chains'])])
    
    # get the dimensions of the plot.
    n_cols = len(indices)
    n_rows = 1
    
    # Plot field and solution.
    fig, axes = plt.subplots(nrows=n_rows, ncols=n_cols, figsize = (8*n_cols, 6))
    
    for i, par_i in enumerate(indices):
        axes[i].set_title('Theta_{}'.format(par_i))
        
        # plot fractal wyrm.
        if plot_type=='trace':
            axes[i].plot(samples[:,par_i], color='c')
        
        # plot histogram.
        elif plot_type=='histogram':
            axes[i].hist(samples[:,par_i], color='c')
        
        # otherwise, plot fractal wyrm.    
        else:
            axes[i].plot(samples[:,par_i], color='c')
    
def plot_sample_matrix(samples, indices=[0,1]):
    
    '''
    Plot a pairs-plot with scatter and kde. The input must be a nxd array 
    or list of nxd arrays, where n is the number of samples  and d is the 
    parameter dimension. Legacy function.
    
    Parameters
    ----------
    samples : dict
        A dict with numpy array(s) containing samples.
    indices : list, optional
        Which parameter indices to plot. The default is [0,1].
    '''

    warnings.warn(' plot_sample_matrix() has been deprecated. Please use to_inference_data() and ArviZ for posterior analysis.')

    samples = np.vstack([samples['chain_{}'.format(i)] for i in range(samples['n_chains'])])
    
    # plug parameters into dataframe and name the columns.
    df_param = pd.DataFrame()
    for i, par_i in enumerate(indices):
        df_param['$\\theta_{{{}}}$'.format(par_i)] = samples[:,par_i]
    
    #cmap = sns.cubehelix_palette(dark=0, light=1.1, rot=-.4, as_cmap=True)
    
    # do a pairs-plot.
    g = sns.PairGrid(df_param)
    g = g.map_upper(sns.scatterplot, size=1.0, color='#19068c', edgecolor='k', alpha=0.1)
    g = g.map_lower(sns.kdeplot, cmap='plasma', shade=False)
    g = g.map_diag(sns.kdeplot, color='#19068c')
    for ax in g.axes[:,0]:
        ax.get_yaxis().set_label_coords(-0.4,0.5)
        
def compute_R_hat(samples, rank_normalised=False, ess_auxiliary=False):
    
    '''
    R-hat according to Vehtari et al. (2020). The first argument (parameters) 
    must be list of arrays, one from each chain. Each array must be nxd, 
    where n is the number of samples and d is the parameter dimension.
    Legacy function.
    
    Parameters
    ----------
    samples : dict
        A dict with numpy array(s) containing samples.
    rank_normalised : bool, optional
        Whether to rank-normalise the samples before computing R-hat. 
        The default is False.
    return_ess_stats : bool, optional
        Whether to return s_m_sq, W and var_hat in place of R-hat. Used internally. 
        The default is False.
    
    Returns
    ----------
    numpy.ndarray
        A numpy array with the R-hat for each parameter.
    '''

    warnings.warn(' compute_R_hat() has been deprecated. Please use to_inference_data() and ArviZ for posterior analysis.')

    if not ess_auxiliary:
        samples = [samples['chain_{}'.format(i)] for i in range(samples['n_chains'])]

    # rank normalise, if the switch is set.
    if rank_normalised:
        samples = rank_normalise(samples)
    else:
        pass
    
    # get the number of chains and number of samples.
    M = len(samples)
    N = samples[0].shape[0]
    
    # compute theta_bar and s_sq for each chain m.
    theta_bar_m = []
    s_m_sq = []
    for i, s in enumerate(samples):
        theta_bar_m.append(s.mean(axis=0))
        s_m_sq.append(1/(N-1) * ((s - theta_bar_m[i])**2).sum(axis=0))
    theta_bar_m = np.array(theta_bar_m)
    s_m_sq = np.array(s_m_sq)
    
    # get the overall theta_bar
    theta_bar = theta_bar_m.mean(axis=0)
    
    # compute between-chain ans within-chain variance.
    B = N/(M-1) * ((theta_bar_m - theta_bar)**2).sum(axis=0)
    W = 1/M * s_m_sq.sum(axis=0)
    
    # compute the marginal posterior variance.
    var_hat = (N-1)/N * W + 1/N * B
    
    # compute r_hat.
    r_hat = np.sqrt(var_hat/W)
    
    if not ess_auxiliary:
        return r_hat
    else:
        return s_m_sq, W, var_hat
    
def compute_ESS(samples, rank_normalised=False):
    
    '''
    ESS according to Vehtari et al. (2020). The first argument (parameters) 
    must be list of arrays, one from each chain. Each array must be nxd, 
    where n is the number of samples and d is the parameter dimension.
    Legacy function.
    
    Parameters
    ----------
    samples : dict
        A dict with numpy array(s) containing samples.
    rank_normalised : bool, optional
        Whether to rank-normalise the samples before computing ESS. 
        The default is False.
    
    Returns
    ----------
    numpy.ndarray
        A numpy array with the ESS for each parameter.
    '''

    warnings.warn(' compute_ESS() has been deprecated. Please use to_inference_data() and ArviZ for posterior analysis.')

    samples = [samples['chain_{}'.format(i)] for i in range(samples['n_chains'])]
    
    # rank normalise, if the switch is set.
    if rank_normalised:
        samples = rank_normalise(samples)
    else:
        pass
    
    # get the number of chains, number of samples, and parameter dimensionality.
    M = len(samples)
    N = samples[0].shape[0]
    n_par = samples[0].shape[1]
    
    # get s_m_sq, W, var_hat from the compute_R_hat function.
    s_m_sq, W, var_hat = compute_R_hat(samples, ess_auxiliary=True)
    
    # set up a list to hold the ESS for each parameter.
    ESS = []
    
    # iterate through the parameters.
    for i in range(n_par):
        
        # set up lists for the autocorrelation and the maximum lag.
        rho_t_m = []
        
        # compute the autocorrelation of the parameter for each chain.
        for j in range(M):
            rho_t_m.append(get_autocorrelation(samples[j][:,i]))
        
        # make sure the autocorrelation of each chain has the same maximum lag.
        rho_t_m = np.array(rho_t_m)
        
        # compute rho_t of the parameter for all the chains.
        rho_t = 1 - (W[i] - 1/M * (s_m_sq[:,i][...,np.newaxis]*rho_t_m).sum(axis=0)) / var_hat[i]
        
        if rho_t.shape[0]%2 == 0:
            P_t = rho_t[::2]+rho_t[1::2]
        else:
            P_t = rho_t[:-1:2]+rho_t[1::2]
        
        k = np.argmax(P_t < 0)
        
        # get the autocorrelation length.
        tau = -1 + 2*P_t[:k].sum()
        
        # compute ESS and append to list.
        ESS.append(N*M/tau)
        
    return np.array(ESS)
    
def get_autocorrelation(x, T=None):
    '''
    Compute the autocorrelation function for a vector x.
     
    Parameters
    ----------
    x : numpy.ndarray
        The vector to compute the autocorrelation for.
    T : int, optional
        The truncation length. Default is None.
    
    Returns
    ----------
    numpy.ndarray
        A numpy array with the autocorrelation for each lag.
    '''

    warnings.warn(' get_autocorrelation() has been deprecated. Please use to_inference_data() and ArviZ for posterior analysis.')

    # get the mean and the length.
    x = x - np.mean(x)
    N = len(x)
    
    # transform to frequency domain.
    fvi = np.fft.fft(x, n=2*N)
    
    # get the autocorrelation curve.
    rho = np.real(np.fft.ifft( fvi * np.conjugate(fvi) )[:N] )
    rho /= N - np.arange(N); rho /= rho[0]
    
    if T is None:
        return rho
    else:
        return rho[:T]
    
def rank_normalise(samples):
    '''
    Rank-normalise samples from multiple chains.
     
    Parameters
    ----------
    samples : list
        A list of numpy arrays, one for each chain.
    
    Returns
    ----------
    list
        List of rank-normalised numpy arrays, one for each chain.
    '''

    warnings.warn(' rank_normalise() has been deprecated. Please use to_inference_data() and ArviZ for posterior analysis.')

    # get the number of samples in each chain.
    n_samples = samples[0].shape[0]
    
    # rank the data and compute the z-transform.
    my_norm = norm()
    r = rankdata(np.vstack(samples), axis=0)
    z = my_norm.ppf((r - 3/8)/(r.shape[0] - 1/4))
    
    # redistribute into the original chain-structure.
    z_list = []
    for i in range(len(samples)):
        z_list.append(z[i*n_samples:(i+1)*n_samples,:])
    
    return z_list
    

def compute_ESS_wolff(x):
    
    '''
    Legacy ESS function for a single MCMC chain. 
    See Ulli Wolff (2006): Monte Carlo errors with less errors.
    '''

    warnings.warn(' compute_ESS_wolff() has been deprecated. Please use to_inference_data() and ArviZ for posterior analysis.')

    N = len(x)
        
    x_mean = np.mean(x)
    t_max = int(np.floor(N/2))
    S_tau = 1.5
    
    time_vec = np.zeros(N+t_max)
    
    for i in range(len(x)):
        time_vec[i] = x[i] - x_mean
    
    freq_vec = np.fft.fft(time_vec)
    
    # Compute out1*conj(out2) and store in out1 (x+yi)*(x-yi)
    for i in range(N + t_max):
        freq_vec[i] = np.real(freq_vec[i])**2 + np.imag(freq_vec[i])**2
    
    # Now compute the inverse fft to get the autocorrelation (stored in timeVec)
    time_vec = np.fft.ifft(freq_vec)
    
    for i in range(t_max+1):
        time_vec[i] = np.real(time_vec[i])/(N - i)
    
    # The following loop uses ideas from "Monte Carlo errors with less errors." by Ulli Wolff 
    # to figure out how far we need to integrate
    G_int = 0.0
    W_opt = 0
    
    for i in range(1, t_max+1):
        G_int += np.real(time_vec[i]) / np.real(time_vec[0])
        
        if G_int <= 0:
            tau_W = 1e-15
        else:
            tau_W = S_tau / np.log((G_int+1) / G_int)
        
        g_W = np.exp(-i/tau_W) - tau_W/np.sqrt(i*N)
        
        if g_W < 0:
            W_opt = i
            t_max = min([t_max, 2*i])
            break
    
    # Correct for bias
    CFbb_opt = np.real(time_vec[0])
    
    for i in range(W_opt + 1):
        CFbb_opt += 2*np.real(time_vec[i+1])
    
    CFbb_opt = CFbb_opt/N
    
    for i in range(t_max+1):
        time_vec[i] += CFbb_opt
    
    scale = np.real(time_vec[0])
    
    for i in range(W_opt):
        time_vec[i] = np.real(time_vec[i])/scale
    
    tau_int = 0.0    
    for i in range(W_opt):
        tau_int += np.real(time_vec[i])    
    tau_int -= 0.5
    
    frac = min([1.0, 1/(2*tau_int)])
    
    return N*frac
