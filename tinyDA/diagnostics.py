import pickle

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from scipy.stats import norm, rankdata
from scipy.ndimage import convolve
        
def plot_parameters(parameters, indices=[0, 1], burnin=0, plot_type='fractal_wyrm'):
    
    '''
    Plot either fractal wyrm plots (hairy caterpillars) or histograms of MCMC
    parameters, given as a nxd array, where n is the number of samples and d is
    the parameter dimension. The second argument is the parameter indices to plot (list).
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
    is the number of samples and d is the parameter dimension. The second 
    argument is the parameter indices to plot (list).
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
    Second argument is a switch to turn on rank-normalisation.
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
    
def compute_ESS(parameters, bulk=False):
    
    '''
    ESS according to Vehtari et al. (2020). The first argument (parameters) 
    must be list of arrays, one from each chain. Each array must be nxd, 
    where n is the number of samples and d is the parameter dimension.
    Second argument is a switch to turn on rank-normalisation for bulk-ESS.
    '''
    
    # rank normalise, if the switch is set.
    if bulk:
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
        T = []
        
        # compute the autocorrelation of the parameter for each chain.
        for j in range(M):
            rho_t_m.append(get_autocorrelation(parameters[j][:,i]))
            T.append(rho_t_m[j].shape[0])
        
        # make sure the autocorrelation of each chain has the same maximum lag.
        T_min = min(T)
        rho_t_m = np.array([r[:T_min] for r in rho_t_m])
        
        # compute rho_t of the parameter for all the chains.
        rho_t = 1 - (W[i] - 1/M * (s_m_sq[:,i][...,np.newaxis]*rho_t_m).sum(axis=0)) / var_hat[i]
        
        # get the autocorrelation length.
        tau = 1 + 2*rho_t[1:].sum()
        
        # compute ESS and append to list.
        ESS.append(N*M/tau)
        
    return np.array(ESS)
    
def get_autocorrelation(x):
    
    # get the mean and the length.
    x = x - np.mean(x)
    N = len(x)
    
    # transform to frequency domain.
    fvi = np.fft.fft(x, n=2*N)
    
    # get the autocorrelation curve.
    rho = np.real(np.fft.ifft( fvi * np.conjugate(fvi) )[:N] )
    rho /= N - np.arange(N); rho /= rho[0]
    
    # get the positive sequence.
    P = convolve(rho, np.array([1,1]))
    T = np.argmax(np.array(P) < 0)
    rho = rho[:T]
    
    return rho
    
def rank_normalise(parameters):
    
    # rank and compute the z-transform.
    my_norm = norm()
    r = rankdata(np.vstack(parameters), axis=0)
    z = my_norm.ppf((r - 3/8)/(r.shape[0] - 1/4))
    
    # redistribute into the original chain-structure.
    z_list = []
    for i in range(len(parameters)):
        z_list.append(z[i*parameters[0].shape[0]:(i+1)*parameters[0].shape[0],:])
    
    return z_list
    

def compute_ESS_wolff(x):
    
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
