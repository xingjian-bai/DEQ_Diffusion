# %%
from functools import partial
import torch
from utils import *

# %%
class Unet(nn.Module):
    """
    Args:
        dim: the number of channels parameter for the model
        init_dim: the number of channels for the first convolution
        out_dim: the number of channels for the last convolution
        dim_mults: the number of channels for each convolution block
        channels: the number of channels in the input
        with_time_emb: whether to use a time embedding
        resnet_block_groups: the number of groups for the resnet blocks
        use_convnext: whether to use convnext blocks
        convnext_mult: the number of channels for the convnext blocks
    """
    def __init__(
        self,
        cfg
        # img_size,
        # channels,
        # init_dim=None,
        # out_dim=None,
        # dim_mults=(1, 2, 4, 8),
        # with_time_emb=True,
        # resnet_block_groups=8,
        # use_convnext=True,
        # convnext_mult=2,
    ):
        super().__init__()
        self.img_size = cfg.dataset.img_size
        self.n_channels = cfg.dataset.n_channels
        self.batch_size = cfg.dataset.batch_size
        self.with_time_emb = cfg.model.with_time_emb
        self.cfg = cfg['model']
        self.cfg.dim_mults = tuple(self.cfg.dim_mults)

        self.cfg.init_dim = default(self.cfg.init_dim, self.img_size // 3 * 2)
        # print('what is ? ', self.cfg.init_dim, 'because', self.img_size // 3 * 2)
        self.init_conv = nn.Conv2d(self.n_channels, self.cfg.init_dim, 7, padding=3)

        # print('checking ', tuple(self.cfg.dim_mults), type(self.cfg.with_time_emb))
        dims = [self.cfg.init_dim, *map(lambda m: self.img_size * m, self.cfg.dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))
        
        if self.cfg.use_convnext:
            block_klass = partial(ConvNextBlock, mult=self.cfg.convnext_mult)
        else:
            block_klass = partial(ResnetBlock, groups=self.cfg.resnet_block_groups)

        # time embeddings
        if self.with_time_emb:
            time_dim = self.img_size * 4
            self.time_mlp = nn.Sequential(
                SinusoidalPositionEmbeddings(self.img_size),
                nn.Linear(self.img_size, time_dim),
                nn.GELU(),
                nn.Linear(time_dim, time_dim),
            )
        else:
            time_dim = None
            self.time_mlp = None

        # layers
        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])
        num_resolutions = len(in_out)

        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)

            self.downs.append(
                nn.ModuleList(
                    [
                        block_klass(dim_in, dim_out, time_emb_dim=time_dim),
                        block_klass(dim_out, dim_out, time_emb_dim=time_dim),
                        Residual(PreNorm(dim_out, LinearAttention(dim_out))),
                        Downsample(dim_out) if not is_last else nn.Identity(),
                    ]
                )
            )

        mid_dim = dims[-1]
        self.mid_block1 = block_klass(mid_dim, mid_dim, time_emb_dim=time_dim)
        self.mid_attn = Residual(PreNorm(mid_dim, Attention(mid_dim)))
        self.mid_block2 = block_klass(mid_dim, mid_dim, time_emb_dim=time_dim)

        for ind, (dim_in, dim_out) in enumerate(reversed(in_out[1:])):
            is_last = ind >= (num_resolutions - 1)

            self.ups.append(
                nn.ModuleList(
                    [
                        block_klass(dim_out * 2, dim_in, time_emb_dim=time_dim),
                        block_klass(dim_in, dim_in, time_emb_dim=time_dim),
                        Residual(PreNorm(dim_in, LinearAttention(dim_in))),
                        Upsample(dim_in) if not is_last else nn.Identity(),
                    ]
                )
            )

        self.cfg.out_dim = default(self.cfg.out_dim, self.n_channels)
        self.final_conv = nn.Sequential(
            block_klass(self.img_size, self.img_size), nn.Conv2d(self.img_size, self.cfg.out_dim, 1)
        )

    def forward(self, x, time):
        x = self.init_conv(x)

        # time embedding
        t = self.time_mlp(time) if exists(self.time_mlp) else None

        h = []

        # downsample
        for block1, block2, attn, downsample in self.downs:
            x = block1(x, t)
            x = block2(x, t)
            x = attn(x)
            h.append(x)
            x = downsample(x)

        # bottleneck
        x = self.mid_block1(x, t)
        x = self.mid_attn(x)
        x = self.mid_block2(x, t)

        # upsample
        for block1, block2, attn, upsample in self.ups:
            x = torch.cat((x, h.pop()), dim=1)
            x = block1(x, t)
            x = block2(x, t)
            x = attn(x)
            x = upsample(x)

        return self.final_conv(x)

    def __str__(self):
        return ""
# %%
from deq_solvers import FixPointSolver
from deq_core import DEQCore
class DEQ (nn.Module):
    def __init__(self, cfg, 
                #  imgsize, channels, core_type, solver_type, stradegy_type, n_channels = 48, n_inner_channels = 64, kernel_size = 3, num_groups = 8, with_time_emb=True, **kwargs
                 ):
        super().__init__()
        self.img_size = cfg.dataset.img_size
        self.channels = cfg.dataset.n_channels
        self.n_channels = cfg.model.n_channels
        self.n_inner_channels = cfg.model.n_inner_channels
        self.kernel_size = cfg.model.kernel_size

        self.batch_size = cfg.dataset.batch_size
        self.cfg = cfg
        self.with_time_emb = cfg.model.with_time_emb

        
        self.conv1 = nn.Conv2d(self.channels, self.n_channels, kernel_size=self.kernel_size, bias=True, padding=1)
        self.bn1 = nn.BatchNorm2d(self.n_channels)
        self.core = DEQCore(cfg)
        self.fixed_point_solver = FixPointSolver(self.core, cfg)
        self.bn2 = nn.BatchNorm2d(self.n_channels)
        self.conv_back = nn.Conv2d(self.n_channels, self.channels, kernel_size=self.kernel_size, bias=True, padding=1)
        self.avgpool = nn.AvgPool2d(8,8)

        if self.with_time_emb:
            time_dim = self.n_channels
            self.time_mlp = nn.Sequential(
                SinusoidalPositionEmbeddings(self.n_channels),
                nn.Linear(self.n_channels, time_dim),
                nn.GELU(),
                nn.Linear(time_dim, self.n_channels)
            )
        # self.flatten = nn.Flatten()
        # self.fc = nn.Linear(n_channels*4*4,10)
    def forward(self, x, time):
        # time embedding
        t = self.time_mlp(time) if exists(self.time_mlp) else None
        x = self.conv1(x)
        x = self.bn1(x)
        # add two dimensions to the time embedding
        t = t[(..., ) + (None, ) * 2]
        # add time embedding to the input
        x = x + t

        x = self.fixed_point_solver(x)
        x = self.bn2(x)
        x = self.conv_back(x)

        # x = self.flatten(x)
        # x = self.fc(x)
        return x
    
class CentralModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.model_type = cfg.model.type
        if cfg.model.type == 'UNet':
            self.model = Unet(cfg)
        elif cfg.model.type == 'DEQ':
            self.model = DEQ(cfg)
        else:
            raise NotImplementedError
    def forward(self, x, time):
        return self.model(x, time)
    
    def __str__(self):
        return self.model_type