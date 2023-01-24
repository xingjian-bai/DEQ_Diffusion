# %%
import torch
from utils import default
# %%
# forward diffusion
def q_sample(x_start, scheduler, t, noise=None):
    """ Given a starting point, a time, and optionally some noise, 
        return the weighted average of the starting point and the noise. """

    noise = default(noise, torch.randn_like(x_start))
    # if noise is None:
    #     noise = torch.randn_like(x_start)

    sqrt_alphas_cumprod_t = scheduler.get_sqrt_alphas_cumprod(t, x_start.shape)
    sqrt_one_minus_alphas_cumprod_t = scheduler.get_sqrt_one_minus_alphas_cumprod(t, x_start.shape)
    
    return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise
# %%
def get_noisy_image(x_start, t, reverse_transform_to_image):
  # add noise
  x_noisy = q_sample(x_start, t=t)

  # turn back into PIL image
  noisy_image = reverse_transform_to_image(x_noisy.squeeze())

  return noisy_image

