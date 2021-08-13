import pickle
from itertools import compress

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from scipy.stats import norm, rankdata
from scipy.ndimage import convolve

def get_parameters(chain, level='fine'):
    '''
    Parameters
    ----------
    chain : tinyDA.Chain or tinyDA.DAChain
        A chain object with samples.
    level : str, optional
        Which level to extract samples from. The default is 'fine'.
    
    Returns
    ----------
    list or numpy.ndarray
        A (list of) numpy array(s) with parameters as columns and samples as rows.
    '''
    # if this is a parallel chain, get all the fine chains.
    if hasattr(chain, 'chains'):
        parameters = []
        for c in chain.chains:
            parameters.append(np.array([link.parameters for link in c]))
        return parameters
        
    # if this is a single-level chain, just get the chain.
    elif hasattr(chain, 'chain'):
        links = chain.chain
    else:
        # if it is a delayed acceptance chain, get the specified level.
        if level == 'fine':
            links = chain.chain_fine
        elif level == 'coarse':
            links = compress(chain.chain_coarse, chain.is_coarse)
    
    # return an array of the parameters.
    return np.array([link.parameters for link in links])
    
        
def plot_parameters(parameters, indices=[0, 1], burnin=0, plot_type='fractal_wyrm'):
    '''
    Plot either fractal wyrm plots (traceplots, hairy caterpillars, etc.) 
    or histograms of MCMC parameters, given as a nxd array, where n is 
    the number of samples and d is the parameter dimension.
    
    Parameters
    ----------
    parameters : numpy.ndarray
        A numpy array with parameters as columns and samples as rows.
    indices : list, optional
        Which parameter indices to plot. The default is [0,1].
    burnin : int, optional
        The burnin length. The default is 0.
    plot_type : str, optional
        The plot type. Can be 'fractal_wyrm' (traceplot) or 'histogram'. 
        The default is 'fractal_wyrm'.
    '''
    
    # get the dimensions of the plot.
    n_cols = len(indices)
    n_rows = 1
    
    # Plot field and solution.
    fig, axes = plt.subplots(nrows=n_rows, ncols=n_cols, figsize = (8*n_cols, 6))
    
    for i, par_i in enumerate(indices):
        axes[i].set_title('Theta_{}'.format(par_i))
        
        # plot fractal wyrm.
        if plot_type=='fractal_wyrm':
            axes[i].plot(parameters[burnin:,par_i], color='c')
        
        # plot histogram.
        elif plot_type=='histogram':
            axes[i].hist(parameters[burnin:,par_i], color='c')
        
        # otherwise, plot fractal wyrm.    
        else:
            axes[i].plot(parameters[burnin:,par_i], color='c')
    
def plot_parameter_matrix(parameters, indices=[0,1], burnin=0):
    
    '''
    Plot a pairs-plot with scatter and kde, given as a nxd array, where n 
    is the number of samples and d is the parameter dimension.
    
    Parameters
    ----------
    parameters : numpy.ndarray
        A numpy array with parameters as columns and samples as rows.
    indices : list, optional
        Which parameter indices to plot. The default is [0,1].
    burnin : int, optional
        The burnin length. The default is 0.
    '''
    
    # plug parameters into dataframe and name the columns.
    df_param = pd.DataFrame()
    for i, par_i in enumerate(indices):
        df_param['$\\theta_{}$'.format(par_i)] = parameters[burnin:,par_i]
    
    #cmap = sns.cubehelix_palette(dark=0, light=1.1, rot=-.4, as_cmap=True)
    
    # do a pairs-plot.
    g = sns.PairGrid(df_param)
    g = g.map_upper(sns.scatterplot, size=1.0, color='#19068c', edgecolor='k', alpha=0.1)
    g = g.map_lower(sns.kdeplot, cmap='plasma', shade=False)
    g = g.map_diag(sns.kdeplot, color='#19068c')
    for ax in g.axes[:,0]:
        ax.get_yaxis().set_label_coords(-0.4,0.5)
        
def compute_R_hat(parameters, rank_normalised=False, return_ess_stats=False):
    
    '''
    R-hat according to Vehtari et al. (2020). The first argument (parameters) 
    must be list of arrays, one from each chain. Each array must be nxd, 
    where n is the number of samples and d is the parameter dimension.
    
    Parameters
    ----------
    parameters : list
        A list of numpy arrays, one for each chain.
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

    # rank normalise, if the switch is set.
    if rank_normalised:
        parameters = rank_normalise(parameters)
    else:
        pass
    
    # get the number of chains and number of samples.
    M = len(parameters)
    N = parameters[0].shape[0]
    
    # compute theta_bar and s_sq for each chain m.
    theta_bar_m = []
    s_m_sq = []
    for i, par in enumerate(parameters):
        theta_bar_m.append(par.mean(axis=0))
        s_m_sq.append(1/(N-1) * ((par - theta_bar_m[i])**2).sum(axis=0))
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
    
    if not return_ess_stats:
        return r_hat
    else:
        return s_m_sq, W, var_hat
    
def compute_ESS(parameters, rank_normalised=False):
    
    '''
    ESS according to Vehtari et al. (2020). The first argument (parameters) 
    must be list of arrays, one from each chain. Each array must be nxd, 
    where n is the number of samples and d is the parameter dimension.
    
    Parameters
    ----------
    parameters : list
        A list of numpy arrays, one for each chain.
    rank_normalised : bool, optional
        Whether to rank-normalise the samples before computing ESS. 
        The default is False.
    
    Returns
    ----------
    numpy.ndarray
        A numpy array with the ESS for each parameter.
    '''
    
    # rank normalise, if the switch is set.
    if rank_normalised:
        parameters = rank_normalise(parameters)
    else:
        pass
    
    # get the number of chains, number of samples, and parameter dimensionality.
    M = len(parameters)
    N = parameters[0].shape[0]
    n_par = parameters[0].shape[1]
    
    # get s_m_sq, W, var_hat from the compute_R_hat function.
    s_m_sq, W, var_hat = compute_R_hat(parameters, return_ess_stats=True)
    
    # set up a list to hold the ESS for each parameter.
    ESS = []
    
    # iterate through the parameters.
    for i in range(n_par):
        
        # set up lists for the autocorrelation and the maximum lag.
        rho_t_m = []
        
        # compute the autocorrelation of the parameter for each chain.
        for j in range(M):
            rho_t_m.append(get_autocorrelation(parameters[j][:,i]))
        
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
    
def rank_normalise(parameters):
    '''
    Rank-normalise samples from multiple chains.
     
    Parameters
    ----------
    parameters : list
        A list of numpy arrays, one for each chain.
    
    Returns
    ----------
    list
        List of rank-normalised numpy arrays, one for each chain.
    '''
    
    # get the number of samples in each chain.
    n_samples = parameters[0].shape[0]
    
    # rank the data and compute the z-transform.
    my_norm = norm()
    r = rankdata(np.vstack(parameters), axis=0)
    z = my_norm.ppf((r - 3/8)/(r.shape[0] - 1/4))
    
    # redistribute into the original chain-structure.
    z_list = []
    for i in range(len(parameters)):
        z_list.append(z[i*n_samples:(i+1)*n_samples,:])
    
    return z_list
    

def compute_ESS_wolff(x):
    
    '''
    Legacy ESS function for a single MCMC chain. 
    See Ulli Wolff (2006): Monte Carlo errors with less errors.
    '''
    
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
