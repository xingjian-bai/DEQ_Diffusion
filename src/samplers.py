# %%
import torch
from tqdm import tqdm
# %%
@torch.no_grad()
def p_sample(model, scheduler, x, t, t_index):
    """ Given a model, a starting point, a time, and optionally some noise,
        calculate the model mean,
        then sample from the posterior distribution given by the model mean."""
    betas_t = scheduler.get_betas(t, x.shape)
    sqrt_one_minus_alphas_cumprod_t = scheduler.get_sqrt_one_minus_alphas_cumprod(t, x.shape)
    sqrt_recip_alphas_t = scheduler.get_sqrt_recip_alphas(t, x.shape)
    
    # Equation 11 in the paper
    # Use our model (noise predictor) to predict the mean
    model_mean = sqrt_recip_alphas_t * (
        x - betas_t * model(x, t) / sqrt_one_minus_alphas_cumprod_t
    )

    if t_index == 0:
        return model_mean
    else:
        posterior_variance_t = scheduler.get_posterior_variance(t, x.shape)
        noise = torch.randn_like(x)
        # Algorithm 2 line 4:
        return model_mean + torch.sqrt(posterior_variance_t) * noise 

# Algorithm 2 but save all images:
@torch.no_grad()
def p_sample_loop(model, scheduler, shape):
    """ Given a model, sample backwards in time and restore the image."""
    device = next(model.parameters()).device

    b = shape[0]
    # start from pure noise (for each example in the batch)
    img = torch.randn(shape, device=device)
    imgs = []
    
    for i in tqdm(reversed(range(0, scheduler.timesteps)), desc='sampling loop time step', total=scheduler.timesteps):
        img = p_sample(model, scheduler, img, torch.full((b,), i, device=device, dtype=torch.long), i)
        imgs.append(img.cpu().numpy())
    return imgs

@torch.no_grad()
def sample(model, scheduler, image_size, batch_size=16, channels=3):
    """ For visualization"""
    return p_sample_loop(model, scheduler, shape=(batch_size, channels, image_size, image_size))