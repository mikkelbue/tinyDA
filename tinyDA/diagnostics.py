####################
# This is code from a different project that I have not adapted yet.
# Please do not use.
####################
import pickle

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

class ChainDiagnostics:
    '''
    This is code from a different project that I have not adapted yet.
    Please do not use.
    '''
    
    def __init__(self, chain):
        self.chain = chain
        
        self.prior = np.zeros(len(self.chain))
        self.likelihood = np.zeros(len(self.chain))
        self.posterior = np.zeros(len(self.chain))
        
        self.parameters = np.zeros((len(self.chain), self.chain[0].parameters.shape[0]))
        
        self.qoi = np.zeros(len(self.chain))
        
        for i, link_i in enumerate(self.chain):
            self.prior[i] = link_i.prior
            self.likelihood[i] = link_i.likelihood
            self.posterior[i] = link_i.posterior
            self.parameters[i,:] = link_i.parameters
            self.qoi[i] = link_i.qoi
            
    def get_mean(self, burnin=0):
        return self.parameters[burnin:].mean(axis=0)
            
    def plot_statistics(self, burnin=0):
        # Plot field and solution.
        fig, axes = plt.subplots(nrows=1, ncols=3, figsize = (24, 6))
        
        axes[0].set_title('Prior')
        axes[0].plot(self.prior[burnin:], color='y') 
        
        axes[1].set_title('Likelihood')
        axes[1].plot(self.likelihood[burnin:], color='c')  
        
        axes[2].set_title('Posterior')
        axes[2].plot(self.posterior[burnin:], color='m')
        
    def plot_parameters(self, parameters, burnin=0, plot_type='fractal_wyrm'):
        
        n_cols = len(parameters)
        n_rows = 1
        
        # Plot field and solution.
        fig, axes = plt.subplots(nrows=n_rows, ncols=n_cols, figsize = (8*n_cols, 6))
        
        for i, par_i in enumerate(parameters):
            axes[i].set_title('Theta_{}'.format(par_i))
            if plot_type=='fractal_wyrm':
                axes[i].plot(self.parameters[burnin:,par_i], color='c')
            elif plot_type=='histogram':
                axes[i].hist(self.parameters[burnin:,par_i], color='c')
            else:
                axes[i].plot(self.parameters[burnin:,par_i], color='c')
                
    def plot_qoi(self, burnin=0, plot_type='histogram'):
        plt.title('QoI')
        if plot_type=='fractal_wyrm':
            plt.plot(self.qoi[burnin:], color='c')
        elif plot_type=='histogram':
            plt.hist(self.qoi[burnin:], color='c')
        else:
            plt.hist(self.qoi[burnin:], color='c')
        plt.show()
        
    def plot_parameter_matrix(self,  parameters, burnin=0):
        
        df_param = pd.DataFrame()
        for i in parameters:
            df_param['$\\theta_{}$'.format(i)] = self.parameters[burnin:,i]
        
        #cmap = sns.cubehelix_palette(dark=0, light=1.1, rot=-.4, as_cmap=True)
        
        g = sns.PairGrid(df_param)
        g = g.map_upper(sns.scatterplot, size=1.0, color='#19068c', edgecolor='k', alpha=0.1)
        g = g.map_lower(sns.kdeplot, cmap='plasma', shade=False)
        g = g.map_diag(sns.kdeplot, color='#19068c')
        for ax in g.axes[:,0]:
            ax.get_yaxis().set_label_coords(-0.4,0.5)
        
    def compute_ESS(self, burnin=0):
        
        qoi = self.qoi[burnin:]
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
        
    def save(self, filename):
        with open(filename, 'wb') as f:
            pickle.dump(self, f)

def load_diagnostics(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)
