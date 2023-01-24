# %%
from forward import q_sample
import torch.nn.functional as F
import torch
# %%
def p_losses(denoise_model, x_start, t, noise=None, loss_type="l1"):
    """ Given a denoising model, a starting point, a time, and optionally some noise,
        return the loss of the denoising model on the noisy image."""
    if noise is None:
        noise = torch.randn_like(x_start)

    x_noisy = q_sample(x_start=x_start, t=t, noise=noise)
    predicted_noise = denoise_model(x_noisy, t)

    if loss_type == 'l1':
        loss = F.l1_loss(noise, predicted_noise)
    elif loss_type == 'l2':
        loss = F.mse_loss(noise, predicted_noise)
    elif loss_type == "huber":
        loss = F.smooth_l1_loss(noise, predicted_noise)
    else:
        raise NotImplementedError()

    return loss