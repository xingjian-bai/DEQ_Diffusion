#%%
from deq_solvers import DEQFixedPointSolver, Solver
from deq_core import ResNetLayer
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
from utils import *

# %%

class DEQ (nn.Module):
    def __init__(self, solver_type, channels = 3, n_channels = 48, n_inner_channels = 64, kernel_size = 3, num_groups = 8, **kwargs):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, n_channels, kernel_size=3, bias=True, padding=1)
        self.bn1 = nn.BatchNorm2d(n_channels)
        self.core = ResNetLayer(n_channels, n_inner_channels, kernel_size, num_groups)
        self.solver = Solver(solver_type)
        self.fixed_point_solver = DEQFixedPointSolver(self.core, self.solver, **kwargs)
        self.bn2 = nn.BatchNorm2d(n_channels)
        self.conv_back = nn.Conv2d(n_channels, channels, kernel_size=3, bias=True, padding=1)
        self.avgpool = nn.AvgPool2d(8,8)
        # self.flatten = nn.Flatten()
        # self.fc = nn.Linear(n_channels*4*4,10)
    def forward(self, x, time):
        # time embedding
        # t = self.time_mlp(time) if exists(self.time_mlp) else None

        # print('input size is: ', x.size())
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.fixed_point_solver(x)
        x = self.bn2(x)
        # x = self.avgpool(x)
        x = self.conv_back(x)

        # print('output size is: ', x.size())
        # x = self.flatten(x)
        # x = self.fc(x)
        return x
# %%
