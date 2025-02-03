import numpy as np

import matplotlib.pyplot as plt

from scipy.spatial import distance_matrix
from scipy.linalg import eigh

class RandomProcess:
    '''
    This class sets up a random process on a grid and generates a realisation of the process, 
    given parameters a random vector. It has no constructor method, since that will
    vary from kernel to kernel.
    '''
    
    def plot_covariance_matrix(self):
        
        # Plot the covariance matrix.
        plt.figure(figsize = (10,8)); plt.imshow(self.cov, cmap = 'binary'); plt.colorbar(); plt.show()
    
    def compute_eigenpairs(self, mkl):
        
        self.mkl = mkl
        
        # Find eigenvalues and eigenvectors using Arnoldi iteration.
        eigvals, eigvecs = eigh(self.cov, eigvals = (self.cov.shape[0] - self.mkl, self.cov.shape[0] - 1))
        
        # Sort the eigenvalues in descending order.
        order = np.flip(np.argsort(eigvals))
        self.eigenvalues = eigvals[order]
        self.eigenvectors = eigvecs[:,order]
      
    def generate(self, coefficients=None, mean=0.0, stdev=1.0):
        
        # Generate a random field, see
        # Scarth, C., Adhikari, S., Cabral, P. H., Silva, G. H. C., & Prado, A. P. do. (2019). 
        # Random field simulation over curved surfaces: Applications to computational structural mechanics. 
        # Computer Methods in Applied Mechanics and Engineering, 345, 283–301. https://doi.org/10.1016/j.cma.2018.10.026
        
        if coefficients is None:
            coefficients = np.random.normal(size=self.mkl)
        
        # Compute the random field.
        self.random_field = mean + stdev*np.linalg.multi_dot((self.eigenvectors, 
                                                              np.diag(np.sqrt(self.eigenvalues)), 
                                                              coefficients))
    

class SquaredExponential(RandomProcess):
    def __init__(self, points, lamb):
        
        '''
        This class inherits from RandomProcess and creates an squared exponential covariance matrix.
        '''

        # Create a distance matrix.
        dist = distance_matrix(points, points)
        
        # Compute the squared covariance.
        self.cov =  np.exp(-0.5*(dist/lamb)**2)

class ARD_Squared(RandomProcess):
    def __init__(self, points, lamb):
        
        '''
        This class inherits from RandomProcess and creates an ARD squared covariance matrix.
        '''

        # Iterate through the ARD scaled distances
        dist = np.zeros((points.shape[0], points.shape[0]))
        for i, lambda_i in enumerate(lamb):
            dist += (distance_matrix(np.expand_dims(points[:,i], axis =  1), 
                                     np.expand_dims(points[:,i], axis =  1))/lamb[i])**2
        
        # Set up the covariance matrix.
        self.cov = np.exp(-0.5*dist)

class Matern32(RandomProcess):
    def __init__(self, points, lamb):
        
        '''
        This class inherits from RandomProcess and creates a Matern 3/2 covariance matrix.
        '''

        # Compute scaled distances.
        dist = np.sqrt(3)*distance_matrix(points, points)/lamb
        
        # Set up Matern 3/2 covariance matrix.
        self.cov =  (1 + dist) * np.exp(-dist)
        
class Matern52(RandomProcess):
    def __init__(self, points, lamb):
        
        '''
        This class inherits from RandomProcess and creates a Matern 5/2 covariance matrix.
        '''

        # Compute scaled distances.
        dist = np.sqrt(5)*distance_matrix(points, points)/lamb
        
        # Set up Matern 5/2 covariance matrix.
        self.cov =  (1 + dist + dist**2/3) * np.exp(-dist)
