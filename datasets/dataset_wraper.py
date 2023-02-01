class DatasetWraper():
    def __init__(self, cfg):
        self.dataset_name = cfg.dataset.name
        if self.dataset_name == "FasionMNIST":
            from fasion_mnist import FasionMNISTDataset
            self.dataset = FasionMNISTDataset(cfg)
        else:
            raise NotImplementedError
    def DataLoader(self):
        return self.dataset.DataLoader()
    
    # def Dataset(self):
    #     return self.dataset.Dataset()
    
    def TestDataloader(self):
        return self.dataset.TestDataloader()
    
    def visualize (self, model, scheduler, experiment_name, loss, gif_num = 1, side_num = 4):
        return self.dataset.visualize(model, scheduler, experiment_name, loss, gif_num = gif_num, side_num = side_num)
    
    def __str__(self) -> str:
        return self.dataset_name
        