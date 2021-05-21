import numpy as np
import TransportMaps as tm

class DataDist(tm.Distributions.Distribution):    
    def __init__(self, dim, x, w=None):
        super().__init__(dim)
        self.x = x
        self.w = w
        
    def quadrature(self, qtype, qparams, *args, **kwargs):
        if qtype == 0: # Monte-Carlo
            x = self.x
            if self.w is None:
                w = np.ones(x.shape[0])
            else:
                w = self.w
        else: 
            raise ValueError("Quadrature not defined")
        return (x,w)

def get_gaussian_transport_map(data, order=1, reg_alpha=1, initial_guess=None):
    
    # get the dimension
    dim = data.shape[1]
    
    # first, rescale the data with a linear map.
    a = np.array([4*(x.min() + x.max())/(x.min() - x.max()) for x in data.T])
    b = np.array([8./(x.max() - x.min()) for x in data.T])
    L = tm.Maps.FrozenLinearDiagonalTransportMap(a, b)
    
    rho = tm.Distributions.GaussianDistribution(np.zeros(dim), np.eye(dim))
    pi = DataDist(dim, data)
    
    # create our transport map and the push distributions.
    S = tm.Default_IsotropicIntegratedSquaredTriangularTransportMap(dim, order, 'total')
    push_L_pi = tm.Distributions.PushForwardTransportMapDistribution(L, pi)
    push_SL_pi = tm.Distributions.PushForwardTransportMapDistribution(S, push_L_pi)
    
    # set up parameters for KL minimisation and optimise.
    qtype = 0; qparams = 1; reg = {'type': 'L2', 'alpha': reg_alpha}; tol = 1e-3; ders = 2
    push_SL_pi.minimize_kl_divergence(rho, qtype=qtype, qparams=qparams, regularization=reg, tol=tol, ders=ders, x0=initial_guess)
    
    # create a composite map
    SL = tm.Maps.CompositeMap(S, L)
    coeffs = push_SL_pi.coeffs
    
    return SL, coeffs

def get_gaussian_transport_distribution(data, order=1, reg_alpha=1, initial_guess=None):
    
    # get the dimension
    dim = data.shape[1]
    
    # get the reference distribution
    rho = tm.Distributions.GaussianDistribution(np.zeros(dim), np.eye(dim))
    
    # get the transport map
    SL, _ = get_gaussian_transport_map(data, order, reg_alpha, initial_guess)
    
    pullback_dist = tm.Distributions.PullBackTransportMapDistribution(SL, rho)
    
    return pullback_dist
