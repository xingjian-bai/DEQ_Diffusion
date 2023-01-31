import abc
import os
import torch
from samplers import sample
from dataclasses import dataclass

@dataclass
class CleanDataset(abc.ABC):

    @abc.abstractmethod
    def DataLoader(self):
        raise NotImplementedError
    
    # @abc.abstractmethod
    # def Dataset(self):
    #     raise NotImplementedError
    
    def TestDataloader(self):
        raise NotImplementedError

    def visualize (self, gif_num = 1, side_num = 4, name_prefix = "unamed"):
        raise NotImplementedError
    

