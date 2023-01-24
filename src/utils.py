# %%
import math
from inspect import isfunction
from functools import partial

import matplotlib.pyplot as plt
from tqdm.auto import tqdm
from einops import rearrange

import torch
from torch import nn, einsum
import torch.nn.functional as F


# %%
def exists(x):
    """Check if x is not None"""
    return x is not None

def default(val, d):
    """Return val if it exists, otherwise return d"""
    if exists(val):
        return val
    return d() if isfunction(d) else d

class Residual(nn.Module):
    """Residual block"""
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, *args, **kwargs):
        return self.fn(x, *args, **kwargs) + x




# %%
from pathlib import Path

def num_to_groups(num, divisor):
    groups = num // divisor
    remainder = num % divisor
    arr = [divisor] * groups
    if remainder > 0:
        arr.append(remainder)
    return arr