#%%
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd

def anderson(f, x0, anderson_cfg):
    """ Anderson acceleration for fixed point iteration. """
    bsz, d, H, W = x0.shape
    X = torch.zeros(bsz, anderson_cfg.m, d*H*W, dtype=x0.dtype, device=x0.device)
    F = torch.zeros(bsz, anderson_cfg.m, d*H*W, dtype=x0.dtype, device=x0.device)
    X[:,0], F[:,0] = x0.view(bsz, -1), f(x0).view(bsz, -1)
    X[:,1], F[:,1] = F[:,0], f(F[:,0].view_as(x0)).view(bsz, -1)
    
    H = torch.zeros(bsz, anderson_cfg.m+1, anderson_cfg.m+1, dtype=x0.dtype, device=x0.device)
    H[:,0,1:] = H[:,1:,0] = 1
    y = torch.zeros(bsz, anderson_cfg.m+1, 1, dtype=x0.dtype, device=x0.device)
    y[:,0] = 1
    
    res = []
    for k in range(2, anderson_cfg.max_iter):
        n = min(k,anderson_cfg.m)
        G = F[:,:n]-X[:,:n]
        H[:,1:n+1,1:n+1] = torch.bmm(G,G.transpose(1,2)) + anderson_cfg.lam*torch.eye(n, dtype=x0.dtype,device=x0.device)[None]
        xx = torch.linalg.solve(H[:,:n+1,:n+1], y[:,:n+1])
        # print(xx.shape)
        alpha = torch.linalg.solve(H[:,:n+1,:n+1], y[:,:n+1])[:, 1:n+1, 0]   # (bsz x n)
        
        X[:,k%anderson_cfg.m] = anderson_cfg.beta * (alpha[:,None] @ F[:,:n])[:,0] + (1-anderson_cfg.beta)*(alpha[:,None] @ X[:,:n])[:,0]
        F[:,k%anderson_cfg.m] = f(X[:,k%anderson_cfg.m].view_as(x0)).view(bsz, -1)
        res.append((F[:,k%anderson_cfg.m] - X[:,k%anderson_cfg.m]).norm().item()/(1e-5 + F[:,k%anderson_cfg.m].norm().item()))
        if (res[-1] < anderson_cfg.tol):
            # print('break at iter', k)
            break
    # print(f'break in the end max_iter={max_iter}')
    return X[:,k%anderson_cfg.m].view_as(x0), res

#%%
def forward_iteration(f, x0, forward_config): #max_iter=50, tol=1e-2):
    """ Naive fixed point iteration."""
    f0 = f(x0)
    res = []
    for _ in range(forward_config.max_iter):
        x = f0
        f0 = f(x)
        res.append((f0 - x).norm().item() / (1e-5 + f0.norm().item()))
        if (res[-1] < forward_config.tol):
            break
    return f0, res


class FixedPointJacobianSolver(nn.Module):
    def __init__(self, f, cfg):
        super().__init__()
        self.f = f
        self.cfg = cfg
        
    def forward(self, solver, x0):
        # compute forward pass and re-engage autograd tape
        with torch.no_grad():
            # compute fixed point
            solution, self.forward_residue = solver(lambda z : self.f(z, x0), torch.zeros_like(x0), self.cfg["model"]["solver"])
        solution = self.f(solution, x0)
        
        # set up Jacobian vector product (without additional forward calls)
        solution_grad = solution.clone().detach().requires_grad_()
        f_grad = self.f(solution_grad, x0)
        def backward_hook(grad):
            # compute Jacobian vector product with autograd tape
            selfmade_grad, self.backward_res = solver(lambda y : autograd.grad(f_grad, solution_grad, y, retain_graph=True)[0] + grad,
                                               grad, self.cfg["model"]["solver"])
            return selfmade_grad
        
        # register hook so grad contains Jacobian vector product
        if self.training:
            solution.register_hook(backward_hook)
        return solution

class UnrollingPhantomGradientSolver(nn.Module):
    def __init__(self, f, cfg):
        super().__init__()
        self.f = f
        self.cfg = cfg

    def forward(self, solver, x0): # k = 5, _lambda = 0.5):
        with torch.no_grad():
            # gradient-free iterations
            h, self.forward_residue = solver(lambda z : self.f(z, x0), torch.zeros_like(x0), self.cfg["model"]["solver"])
        if self.training:
            for _ in range (self.cfg.model.stradegy.k):
                # gradient-based iterations
                h = (1 - self.cfg.model.stradegy._lambda) * h + self.cfg.model.stradegy._lambda * self.f(h, x0)
        return h
    
# class NeumannPhantomGradientSolver(nn.Module):
#     def __init__(self, f, cfg):
#         super().__init__()
#         self.f = f
#         self.cfg = cfg

#     def forward(self, solver, x0): #, k = 5, _lambda = 0.5):
        
#         with torch.no_grad():
#             # gradient-free iterations
#             h, self.forward_residue = solver(lambda z : self.f(z, x0), torch.zeros_like(x0), self.cfg["model"]["solver"])
#         def phantom_grad (grad):
#             f = (1 - self.cfg.model.stradegy._lambda) * h + self.cfg.model.stradegy._lambda * self.f(h, x0)

#             g_hat = grad
#             for _ in range (self.cfg.model.stradegy.k - 1):
#                 g_hat = grad + autograd.grad(f, h, g_hat)
#             g_out = self.cfg.model.stradegy._lambda * autograd.grad(f, h, grad_outputs=g_hat)
#             return g_out

#         h_grad = h.clone().detach().requires_grad_()
#         if self.training:
#             h_grad.register_hook(phantom_grad)
#         return h_grad

class JacobianFreeSolver(nn.Module):
    def __init__(self, f, cfg):
        super().__init__()
        self.f = f
        self.cfg = cfg

    def forward(self, solver, x0):
        with torch.no_grad():
            # gradient-free iterations
            h, self.forward_residue = solver(lambda z : self.f(z, x0), torch.zeros_like(x0), self.cfg["model"]["solver"])
        # calc gradient for the last step
        h = self.f(h, x0)
        return h
    
#%%
class FixPointSolver(nn.Module):
    """
    Wrapper for fixed point solvers' iterative algorithm.
    and gradient calculation stradegy.
    """
    def __init__(self, f, cfg):
        super().__init__()
        self.f = f
        self.cfg = cfg
        self.solver_type = cfg.model.solver.type
        self.stradegy_type = cfg.model.stradegy.type

        if self.solver_type == 'anderson':
            self.iter = anderson
        elif self.solver_type == 'forward_iteration':
            self.iter = forward_iteration
        else:
            raise NotImplementedError()
        
        
        if self.stradegy_type == 'jacobian_free':
            self.stradegy = JacobianFreeSolver(f, cfg)
        # elif self.stradegy_type == 'neumann_phantom_gradient':
        #     self.stradegy = NeumannPhantomGradientSolver(f, cfg)
        elif self.stradegy_type == 'unrolling_phantom_gradient':
            self.stradegy = UnrollingPhantomGradientSolver(f, cfg)
        elif self.stradegy_type == 'fixed_point_jacobian':
            self.stradegy = FixedPointJacobianSolver(f, cfg)
        else:
            raise NotImplementedError()

    def forward(self, x0, **kwargs):
        solution = self.stradegy(self.iter, x0, **kwargs)
        return solution