#%%
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd

class DEQFixedPoint(nn.Module):
    """DEQ Fixed Point Solver"""
    def __init__(self, f, solver, **kwargs):
        super().__init__()
        self.f = f
        self.solver = solver
        self.kwargs = kwargs
        
    def forward(self, x):
        # compute forward pass and re-engage autograd tape

        with torch.no_grad():
            # compute fixed point
            z, self.forward_res = self.solver(lambda z : self.f(z, x), torch.zeros_like(x), **self.kwargs)
        z = self.f(z,x)
        
        # set up Jacobian vector product (without additional forward calls)
        z0 = z.clone().detach().requires_grad_()
        f0 = self.f(z0,x)
        def backward_hook(grad):
            # compute Jacobian vector product with autograd tape
            g, self.backward_res = self.solver(lambda y : autograd.grad(f0, z0, y, retain_graph=True)[0] + grad,
                                               grad, **self.kwargs)
            return g
        
        # register hook so grad contains Jacobian vector product
        z.register_hook(backward_hook)
        return z

# from torch.autograd import gradcheck
# # run a very small network with double precision, iterating to high precision
# f = ResNetLayer(2,2, num_groups=2).double()
# deq = DEQFixedPoint(f, anderson, tol=1e-10, max_iter=500).double()
# gradcheck(deq, torch.randn(1,2,3,3).double().requires_grad_(), eps=1e-5, atol=1e-3, check_undefined_grad=False)