#%%
import torch.nn.functional as F
from torch import nn

class ResNetLayer(nn.Module):
    """ResNet layer with GroupNorm and ReLU activation"""
    def __init__(self, cfg):
        # n_channels, n_inner_channels, kernel_size=2, num_groups=8):
        """
        Args:
            n_channels: number of input channels
            n_inner_channels: number of channels in the inner convolution
            kernel_size: kernel size of the inner convolution
            num_groups: number of groups for GroupNorm
        """
        super().__init__()
        self.cfg = cfg['model']['core']
        self.conv1 = nn.Conv2d(cfg.model.n_channels, cfg.model.n_inner_channels, cfg.model.kernel_size, padding=cfg.model.kernel_size//2, bias=False)
        self.conv2 = nn.Conv2d(cfg.model.n_inner_channels, cfg.model.n_channels, cfg.model.kernel_size, padding=cfg.model.kernel_size//2, bias=False)
        self.norm1 = nn.GroupNorm(self.cfg.num_groups, cfg.model.n_inner_channels)
        self.norm2 = nn.GroupNorm(self.cfg.num_groups, cfg.model.n_channels)
        self.norm3 = nn.GroupNorm(self.cfg.num_groups, cfg.model.n_channels)
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        
    def forward(self, z, x):
        y = self.norm1(F.relu(self.conv1(z)))
        return self.norm3(F.relu(z + self.norm2(x + self.conv2(y))))
    # we will choose n_channels in the above layer to be smaller than n_inner_channels

class DEQCore(nn.Module):
    """the core of DEQ wrapper"""
    def __init__(self, cfg):
        super().__init__()
        self.core_type = cfg.model.core.type
        if self.core_type == 'resnet':
            self.core = ResNetLayer(cfg)
        else:
            raise NotImplementedError
    def forward(self, z, x):
        return self.core(z, x)