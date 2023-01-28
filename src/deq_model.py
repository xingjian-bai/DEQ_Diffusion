#%%
from deq_solvers import DEQFixedPointSolver, Solver
from deq_core import ResNetLayer
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
from utils import *
from models import SinusoidalPositionEmbeddings

# %%

class DEQ (nn.Module):
    def __init__(self, solver_type, channels = 3, n_channels = 48, n_inner_channels = 64, kernel_size = 3, num_groups = 8, with_time_emb=True, **kwargs):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, n_channels, kernel_size=3, bias=True, padding=1)
        self.bn1 = nn.BatchNorm2d(n_channels)
        self.core = ResNetLayer(n_channels, n_inner_channels, kernel_size, num_groups)
        self.solver = Solver(solver_type)
        self.fixed_point_solver = DEQFixedPointSolver(self.core, self.solver, **kwargs)
        self.bn2 = nn.BatchNorm2d(n_channels)
        self.conv_back = nn.Conv2d(n_channels, channels, kernel_size=3, bias=True, padding=1)
        self.avgpool = nn.AvgPool2d(8,8)

        if with_time_emb:
            time_dim = n_channels
            self.time_mlp = nn.Sequential(
                SinusoidalPositionEmbeddings(n_channels),
                nn.Linear(n_channels, time_dim),
                nn.GELU(),
                nn.Linear(time_dim, n_channels)
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
# %%
