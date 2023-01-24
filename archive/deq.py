import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class ResNetLayer(nn.Module):
    def __init__(self, n_channels, n_inner_channels, kernel_size=3, num_groups=8):
        super().__init__()
        self.conv1 = nn.Conv2d(n_channels, n_inner_channels, kernel_size, padding=kernel_size//2, bias=False)
        self.conv2 = nn.Conv2d(n_inner_channels, n_channels, kernel_size, padding=kernel_size//2, bias=False)
        self.norm1 = nn.GroupNorm(num_groups, n_inner_channels)
        self.norm2 = nn.GroupNorm(num_groups, n_channels)
        self.norm3 = nn.GroupNorm(num_groups, n_channels)
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        
    def forward(self, z, x):
        y = self.norm1(F.relu(self.conv1(z)))
        y = self.norm3(F.relu(z + self.norm2(x + self.conv2(y))))
        print(f'ResNetLayer shape comparison: {z.shape=} vs {y.shape=} vs {x.shape=}')
        return y
    # we will choose n_channels in the above layer to be smaller than n_inner_channels

import torch.autograd as autograd

class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings

class DEQFixedPoint(nn.Module):
    def __init__(self, f, solver, **kwargs):
        super().__init__()
        image_channels = 3
        out_dim = 1 
        time_emb_dim = 32

        self.f = f
        self.solver = solver
        self.kwargs = kwargs

        self.time_mlp = nn.Sequential(
                SinusoidalPositionEmbeddings(time_emb_dim),
                nn.Linear(time_emb_dim, time_emb_dim),
                nn.ReLU()
            )

        # Initial projection
        # self.conv0 = nn.Conv2d(image_channels, down_channels[0], 3, padding=1)
        # self.output = nn.Conv2d(, 3, out_dim)
        
    def forward(self, x, timestep):
        # Embedd time
        t = self.time_mlp(timestep)

        print(f'shapes: {x.shape=} vs {t.shape=}')


        # # compute forward pass and re-engage autograd tape
        # with torch.no_grad():
        #     z, self.forward_res = self.solver(lambda z : self.f(z, x), torch.zeros_like(x), **self.kwargs)
        # z = self.f(z,x)
        
        # # set up Jacobian vector product (without additional forward calls)
        # z0 = z.clone().detach().requires_grad_()
        # f0 = self.f(z0,x)
        # def backward_hook(grad):
        #     g, self.backward_res = self.solver(lambda y : autograd.grad(f0, z0, y, retain_graph=True)[0] + grad,
        #                                        grad, **self.kwargs)
        #     return g
        # z.register_hook(backward_hook)

        # print(f'whoo to merge? {z.shape}', {time_emb.shape})


        return z
    
