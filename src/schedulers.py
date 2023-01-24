# %%
import torch
import torch.nn.functional as F
from utils import *
# %%
def cosine_beta_schedule(timesteps, s=0.008):
    """
    cosine schedule as proposed in https://arxiv.org/abs/2102.09672
    """
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0.0001, 0.9999)

def linear_beta_schedule(timesteps):
    beta_start = 0.0001
    beta_end = 0.02
    return torch.linspace(beta_start, beta_end, timesteps)

def quadratic_beta_schedule(timesteps):
    beta_start = 0.0001
    beta_end = 0.02
    return torch.linspace(beta_start**0.5, beta_end**0.5, timesteps) ** 2

def sigmoid_beta_schedule(timesteps):
    beta_start = 0.0001
    beta_end = 0.02
    betas = torch.linspace(-6, 6, timesteps)
    return torch.sigmoid(betas) * (beta_end - beta_start) + beta_start
# %%
from utils import extract
class Scheduler:
    def __init__(self, scheduler_type, timesteps = 200):
        self.timesteps = timesteps

        # define beta schedule
        if scheduler_type == "cosine":
            self.betas = cosine_beta_schedule(timesteps=timesteps)
        elif scheduler_type == "linear":
            self.betas = linear_beta_schedule(timesteps=timesteps)
        elif scheduler_type == "quadratic":
            self.betas = quadratic_beta_schedule(timesteps=timesteps)
        elif scheduler_type == "sigmoid":
            self.betas = sigmoid_beta_schedule(timesteps=timesteps)
        else:
            raise NotImplementedError()

        # define alphas 
        self.alphas = 1. - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, axis=0)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)
        self.sqrt_recip_alphas = torch.sqrt(1.0 / self.alphas)

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        self.posterior_variance = self.betas * (1. - self.alphas_cumprod_prev) / (1. - self.alphas_cumprod)
    
    def get_betas(self, t, x_shape):
        return extract(self.betas, t, x_shape)
    def get_alphas(self, t, x_shape):
        return extract(self.alphas, t, x_shape)
    def get_alphas_cumprod(self, t, x_shape):
        return extract(self.alphas_cumprod, t, x_shape)
    def get_alphas_cumprod_prev(self, t, x_shape):
        return extract(self.alphas_cumprod_prev, t, x_shape)
    def get_sqrt_recip_alphas(self, t, x_shape):
        return extract(self.sqrt_recip_alphas, t, x_shape)
    def get_sqrt_alphas_cumprod(self, t, x_shape):
        return extract(self.sqrt_alphas_cumprod, t, x_shape)
    def get_sqrt_one_minus_alphas_cumprod(self, t, x_shape):
        return extract(self.sqrt_one_minus_alphas_cumprod, t, x_shape)
    def get_posterior_variance(self, t, x_shape):
        return extract(self.posterior_variance, t, x_shape)


# %%
