####################
# This is code from a different project that I have not adapted yet.
# Please do not use.
####################
import pickle

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
        
def plot_parameters(parameters, indices=[0, 1], burnin=0, plot_type='fractal_wyrm'):
    
    n_cols = len(indices)
    n_rows = 1
    
    # Plot field and solution.
    fig, axes = plt.subplots(nrows=n_rows, ncols=n_cols, figsize = (8*n_cols, 6))
    
    for i, par_i in enumerate(indices):
        axes[i].set_title('Theta_{}'.format(par_i))
        if plot_type=='fractal_wyrm':
            axes[i].plot(parameters[burnin:,par_i], color='c')
        elif plot_type=='histogram':
            axes[i].hist(parameters[burnin:,par_i], color='c')
        else:
            axes[i].plot(parameters[burnin:,par_i], color='c')
    
def plot_parameter_matrix(parameters, indices=[0,1], burnin=0):
    
    df_param = pd.DataFrame()
    for i, par_i in enumerate(indices):
        df_param['$\\theta_{}$'.format(par_i)] = parameters[burnin:,par_i]
    
    #cmap = sns.cubehelix_palette(dark=0, light=1.1, rot=-.4, as_cmap=True)
    
    g = sns.PairGrid(df_param)
    g = g.map_upper(sns.scatterplot, size=1.0, color='#19068c', edgecolor='k', alpha=0.1)
    g = g.map_lower(sns.kdeplot, cmap='plasma', shade=False)
    g = g.map_diag(sns.kdeplot, color='#19068c')
    for ax in g.axes[:,0]:
        ax.get_yaxis().set_label_coords(-0.4,0.5)
    
def compute_ESS(qoi, burnin=0):
    
    qoi = qoi[burnin:]
    N = len(qoi)
        
    qoi_mean = np.mean(qoi)
    t_max = int(np.floor(N/2))
    S_tau = 1.5
    
    time_vec = np.zeros(N+t_max)
    
    for i in range(len(qoi)):
        time_vec[i] = qoi[i] - qoi_mean
    
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
