
from dataset import CleanDataset

class DatasetWraper(CleanDataset):
    def __init__(self, cfg):
        self.dataset_name = cfg.dataset.name
        if self.dataset_name == "FasionMNIST":
            from fasion_mnist import FasionMNISTDataset
            self.dataset = FasionMNISTDataset()
        else:
            raise NotImplementedError
    def DataLoader(self):
        return self.dataset.DataLoader()
    
    # def Dataset(self):
    #     return self.dataset.Dataset()
    
    def TestDataloader(self):
        return self.dataset.TestDataloader()
    
    def visualize (self, gif_num = 1, side_num = 4, name_prefix = "unamed"):
        return self.dataset.visualize(gif_num = gif_num, side_num = side_num, name_prefix = name_prefix)
    
    def __str__(self) -> str:
        return self.dataset_name
        