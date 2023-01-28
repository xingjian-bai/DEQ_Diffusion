#%%
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd

def anderson(f, x0, m=5, lam=1e-4, max_iter=50, tol=1e-2, beta = 1.0):
    """ Anderson acceleration for fixed point iteration. """
    bsz, d, H, W = x0.shape
    X = torch.zeros(bsz, m, d*H*W, dtype=x0.dtype, device=x0.device)
    F = torch.zeros(bsz, m, d*H*W, dtype=x0.dtype, device=x0.device)
    X[:,0], F[:,0] = x0.view(bsz, -1), f(x0).view(bsz, -1)
    X[:,1], F[:,1] = F[:,0], f(F[:,0].view_as(x0)).view(bsz, -1)
    
    H = torch.zeros(bsz, m+1, m+1, dtype=x0.dtype, device=x0.device)
    H[:,0,1:] = H[:,1:,0] = 1
    y = torch.zeros(bsz, m+1, 1, dtype=x0.dtype, device=x0.device)
    y[:,0] = 1
    
    res = []
    for k in range(2, max_iter):
        n = min(k, m)
        G = F[:,:n]-X[:,:n]
        H[:,1:n+1,1:n+1] = torch.bmm(G,G.transpose(1,2)) + lam*torch.eye(n, dtype=x0.dtype,device=x0.device)[None]
        xx = torch.linalg.solve(H[:,:n+1,:n+1], y[:,:n+1])
        # print(xx.shape)
        alpha = torch.linalg.solve(H[:,:n+1,:n+1], y[:,:n+1])[:, 1:n+1, 0]   # (bsz x n)
        
        X[:,k%m] = beta * (alpha[:,None] @ F[:,:n])[:,0] + (1-beta)*(alpha[:,None] @ X[:,:n])[:,0]
        F[:,k%m] = f(X[:,k%m].view_as(x0)).view(bsz, -1)
        res.append((F[:,k%m] - X[:,k%m]).norm().item()/(1e-5 + F[:,k%m].norm().item()))
        if (res[-1] < tol):
            # print('break at iter', k)
            break
    # print(f'break in the end max_iter={max_iter}')
    return X[:,k%m].view_as(x0), res

#%%
def forward_iteration(f, x0, max_iter=50, tol=1e-2):
    f0 = f(x0)
    res = []
    for k in range(max_iter):
        x = f0
        f0 = f(x)
        res.append((f0 - x).norm().item() / (1e-5 + f0.norm().item()))
        if (res[-1] < tol):
            break
    return f0, res

#%%
class Solver:
    def __init__(self, solver_type):
        if solver_type == 'anderson':
            self.iter = anderson
        elif solver_type == 'forward_iteration':
            self.iter = forward_iteration
        else:
            raise NotImplementedError()
    def __call__(self, *args, **kwargs):
        return self.iter(*args, **kwargs)
    

class DEQFixedPointSolver(nn.Module):
    def __init__(self, f, solver, **kwargs):
        super().__init__()
        self.f = f
        self.solver = solver
        self.kwargs = kwargs
        
    def forward(self, x0):
        # compute forward pass and re-engage autograd tape
        with torch.no_grad():
            # compute fixed point
            solution, self.forward_residue = self.solver(lambda z : self.f(z, x0), torch.zeros_like(x0), **self.kwargs)
        solution = self.f(solution, x0)
        
        # set up Jacobian vector product (without additional forward calls)
        solution_grad = solution.clone().detach().requires_grad_()
        f_grad = self.f(solution_grad, x0)
        def backward_hook(grad):
            # compute Jacobian vector product with autograd tape
            selfmade_grad, self.backward_res = self.solver(lambda y : autograd.grad(f_grad, solution_grad, y, retain_graph=True)[0] + grad,
                                               grad, **self.kwargs)
            return selfmade_grad
        
        # register hook so grad contains Jacobian vector product
        if solution.requires_grad:
            solution.register_hook(backward_hook)
        return solution
