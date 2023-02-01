import torch
from samplers import sample
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import os
from torchvision.transforms import Compose
from torchvision import transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import wandb

# fasion_mnist_transform = Compose([
#             transforms.RandomHorizontalFlip(),
#             transforms.ToTensor(),
#             transforms.Lambda(lambda t: (t * 2) - 1) # normalize to [-1, 1]
# ])

# mnist_transform = Compose([
#             transforms.ToTensor(),
#             transforms.Lambda(lambda t: (t * 2) - 1)
# ])

def fasion_mnist_transforms(examples):
    fasion_mnist_transform = Compose([
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Lambda(lambda t: (t * 2) - 1) # normalize to [-1, 1]
    ])
    examples["pixel_values"] = [fasion_mnist_transform(image.convert("L")) for image in examples["image"]]
    del examples["image"]
    return examples
def mnist_transforms(examples):
    mnist_transform = Compose([
            transforms.ToTensor(),
            transforms.Lambda(lambda t: (t * 2) - 1) # normalize to [-1, 1]
    ])
    examples["pixel_values"] = [mnist_transform(image.convert("L")) for image in examples["image"]]
    del examples["image"]
    return examples

class DatasetWraper():
    def __init__(self, cfg):
        self.cfg = cfg
        if cfg.dataset.name == "FasionMNIST":
            self.transform = fasion_mnist_transforms
        elif cfg.dataset.name == "MNIST":
            self.transform = mnist_transforms
        else:
            raise NotImplementedError
        
        self.dataset = torch.load(cfg.dataset.address)
        self.transformed_dataset = self.dataset.with_transform(self.transform).remove_columns("label")
        self.dataloader = DataLoader(self.transformed_dataset["train"], batch_size=cfg.dataset.batch_size, shuffle=True)
        self.test_dataloader = DataLoader(self.transformed_dataset["test"], batch_size=cfg.dataset.batch_size, shuffle=True)
        # print(f'finish init {cfg.dataset.name} dataset')
        # print(f'size of the transformed dataset: {len(self.transformed_dataset["train"])}')
        # print(f'size of the transformed dataset: {len(self.transformed_dataset["test"])}')
        # print(f'shape of the transformed dataset: {self.transformed_dataset["train"][0][0].shape}')
        # print(f'a sample of the transformed dataset: {self.transformed_dataset["train"][0][0]}')

    def DataLoader(self):
        return self.dataloader
    
    # def Dataset(self):
    #     return self.dataset.Dataset()
    
    def TestDataloader(self):
        return self.test_dataloader
    def make_animation(self, samples, index, experiment_name, epoch):
        fig = plt.figure()
        ims = []
        for i in range(self.cfg.scheduler.timesteps):
            im = plt.imshow(samples[i][index].reshape(self.cfg.dataset.img_size, self.cfg.dataset.img_size, self.cfg.dataset.n_channels), cmap="gray", animated=True)
            ims.append([im])
        animate = animation.ArtistAnimation(fig, ims, interval=50, blit=True, repeat_delay=3000)
        animate.save(f'../gifs/{experiment_name}__ep{epoch}.gif')
        wandb.log({'gif': wandb.Video(f'../gifs/{experiment_name}__ep{epoch}.gif'), 'epoch': epoch})

    def visualize (self, model, scheduler, experiment_name, epoch, loss, gif_num = 1, side_num = 4):
        samples = sample(model, scheduler, image_size=self.cfg.dataset.img_size, batch_size = side_num * side_num, channels=self.cfg.dataset.n_channels)
        _, axarr = plt.subplots(side_num, side_num)
        for i in range(side_num):
            for j in range(side_num):
                axarr[i,j].imshow(samples[-1][i * side_num + j].reshape(self.cfg.dataset.img_size, self.cfg.dataset.img_size, self.cfg.dataset.n_channels), cmap="gray")
        
        plt.suptitle(f'{loss=:.4f}', fontsize = 14)
        plt.tight_layout()
        plt.savefig(f'../gifs/{experiment_name}__ep{epoch}.png')
        wandb.log({'image': wandb.Image(f'../gifs/{experiment_name}__ep{epoch}.png'), 'epoch': epoch})

        for i in range(gif_num):
            self.make_animation(samples, i, experiment_name, epoch)

    def __str__(self) -> str:
        return self.dataset_name
        