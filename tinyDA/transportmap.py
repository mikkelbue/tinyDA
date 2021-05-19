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
