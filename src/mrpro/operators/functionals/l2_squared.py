

import torch
import torch.nn.functional as F  # noqa: N812

from mrpro.operators._Functional import ProximableFunctional

class L2NormSquared(ProximableFunctional):
    
    def __init__(self, lam=1, g:torch.Tensor=None):
        super().__init__(lam=1)
        self.g = g

    def forward(self, x:torch.Tensor) -> torch.Tensor:
        return x.inner(x) # does this work for complex? or should it be vdot
    
    def convex_conj(self):
        return 1/4 * L2NormSquared()
    
    def prox(self, x:torch.Tensor, sigma:torch.Tensor):
        if sigma is torch.scalar_tensor:
            if self.g is None:
                out = x/(1+2*sigma/self.lam)
            else:
                out = x/(1+2*sigma/self.lam)-sigma*self.g/(1+2*sigma/self.lam)
        else:
            if self.g is None:
                out = x/(1+2/self.lam*sigma)
            else:
                out = (x-sigma*self.g)/(1+2/self.lam*sigma)
        
        return out
                
    
    def prox_convex_conj(self, x:torch.Tensor, sigma:torch.Tensor):
        if sigma is torch.scalar_tensor:
            if self.g is None:
                out = x/(1+0.5*sigma/self.lam)
            else:
                out = x/(1+0.5*sigma/self.lam)-sigma*self.g/(1+0.5*sigma/self.lam)
        else:
            if self.g is None:
                out = x/(1+0.5 / self.lam*sigma)
            else:
                out = (x-sigma*self.g)/(1+0.5 / self.lam*sigma)
        
        return out
                