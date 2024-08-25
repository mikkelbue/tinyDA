import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import distance_matrix

class Gravity:
    def __init__(self, depth, n_quad, n_data):
        
        # Set the depth of the density (distance to the surface measurements).
        self.depth = depth
        
        # Set the quadrature degree along one axis.
        self.n_quad = n_quad;
        
        # Set the number of data points along one axis.
        self.n_data = n_data
        
        # Set the quadrature points.
        x = np.linspace(0, 1, self.n_quad+1); tx = (x[1:] + x[:-1]) / 2
        y = np.linspace(0, 1, self.n_quad+1); ty = (y[1:] + y[:-1]) / 2
        self.TX, self.TY = np.meshgrid(tx, ty)
        
        # Set the measurement points.
        x = np.linspace(0, 1, self.n_data+1); sx = (x[1:] + x[:-1]) / 2
        y = np.linspace(0, 1, self.n_data+1); sy = (y[1:] + y[:-1]) / 2
        self.SX, self.SY = np.meshgrid(sx, sy)
        
        # Create coordinate vectors.
        T_coords = np.c_[self.TX.ravel(), self.TY.ravel(), np.zeros(self.n_quad**2)]
        S_coords = np.c_[self.SX.ravel(), self.SY.ravel(), self.depth*np.ones(self.n_data**2)]
        
        # Set the quadrature weights.
        self.w = 1/self.n_quad**2
        
        # Compute a distance matrix
        dist = distance_matrix(S_coords, T_coords)
        
        # Create the Fremholm kernel.
        self.K = self.w * self.depth/dist**3
        
    def geological_model(self, parameters):

        f = np.zeros(self.TX.shape)

        for circle in parameters:
            c = circle['position']; r = circle['radius']
            f += (self.TX - c[0])**2 + (self.TY - c[1])**2 < r**2

        return f.astype(bool).astype(float).flatten()
    
    def solve(self, parameters):
        
        # Internalise the Random Field parameters
        self.parameters = parameters
        
        # Set the density.
        self.f = self.geological_model(self.parameters)
        
        # Compute the signal.
        self.g = np.dot(self.K, self.f)

    def __call__(self, parameters):
        self.solve(parameters)
        return self.g, self.f

    def plot_model(self):
        
        # Plot the density and the signal.
        fig, axes = plt.subplots(1,2, figsize=(16,6))
        axes[0].set_title('Density')
        f = axes[0].imshow(self.f.reshape(self.n_quad, self.n_quad), extent=(0,1,0,1), origin='lower', cmap='plasma')
        fig.colorbar(f, ax=axes[0])
        axes[0].grid(False)
        axes[1].set_title('Noiseless Signal')
        g = axes[1].imshow(self.g.reshape(self.n_data, self.n_data), extent=(0,1,0,1), origin='lower', cmap='plasma')
        fig.colorbar(g, ax=axes[1])
        axes[1].grid(False)
