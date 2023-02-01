# %%
# from torchvision.transforms import Compose, ToTensor, Lambda, ToPILImage, CenterCrop, Resize
# from datasets import load_dataset
# from torchvision import transforms
# from torch.utils.data import DataLoader
# import matplotlib.pyplot as plt
# import torch
# load dataset from the hub
# dataset = load_dataset("fashion_mnist")
# %%
# torch.save(dataset, "../data/fashion_mnist.pkl")
# print('started loading dataset')

# IMPORTANT: current position is in src/output/time/logs

# IMAGE_SIZE = 28
# CHANNELS = 1
# BATCH_SIZE = 128
# %%

# print('finished loading dataset')
# transformed_dataset = dataset.with_transform(transforms).remove_columns("label")
# %%
# # create dataloader
# dataloader = DataLoader(transformed_dataset["train"], batch_size=BATCH_SIZE, shuffle=True)
# test_dataloader = DataLoader(transformed_dataset["test"], batch_size=BATCH_SIZE, shuffle=True)
# print('finished creating dataloader')
# %% save the transformed dataset
# import torch
# torch.save(dataloader, "../data/fashion_mnist_train.pkl")
# torch.save(test_dataloader, "../data/fashion_mnist_test.pkl")
# torch.save(transformed_dataset["train"], "../data/fashion_mnist_train.pkl")
# torch.save(transformed_dataset["test"], "../data/fashion_mnist_test.pkl")

# %%
# import torch
# print('start loading')
# train_data = torch.load("../data/fashion_mnist_train.pkl", )
# test_data = torch.load("../data/fashion_mnist_test.pkl")
# dataloader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
# test_dataloader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=True)
# print('finish loading')
# %%
from samplers import sample
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import os
from torchvision.transforms import Compose
from torchvision import transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import torch
import wandb

transform = Compose([
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Lambda(lambda t: (t * 2) - 1)
])
def transforms(examples):
   examples["pixel_values"] = [transform(image.convert("L")) for image in examples["image"]]
   del examples["image"]
   return examples

# %%
# from dataset import CleanDataset
# class FasionMNISTDataset():
#     def __init__(self, cfg):
#         self.cfg = cfg
#         self.dataset = torch.load("../data/fashion_mnist.pkl")
#         self.transformed_dataset = self.dataset.with_transform(transforms).remove_columns("label")
#         self.dataloader = DataLoader(self.transformed_dataset["train"], batch_size=cfg.dataset.batch_size, shuffle=True)
#         self.test_dataloader = DataLoader(self.transformed_dataset["test"], batch_size=cfg.dataset.batch_size, shuffle=True)
    
#     def DataLoader(self):
#         return self.dataloader
    
#     # def Dataset(self):
#     #     return transformed_dataset
    
#     def TestDataloader(self):
#         return self.test_dataloader
#     def make_animation(self, samples, index, experiment_name, epoch):
#         fig = plt.figure()
#         ims = []
#         for i in range(self.cfg.scheduler.timesteps):
#             im = plt.imshow(samples[i][index].reshape(self.cfg.dataset.img_size, self.cfg.dataset.img_size, self.cfg.dataset.n_channels), cmap="gray", animated=True)
#             ims.append([im])
#         animate = animation.ArtistAnimation(fig, ims, interval=50, blit=True, repeat_delay=3000)
#         animate.save(f'../gifs/{experiment_name}__ep{epoch}.gif')
#         wandb.log({'gif': wandb.Video(f'../gifs/{experiment_name}__ep{epoch}.gif'), 'epoch': epoch})

#     def visualize (self, model, scheduler, experiment_name, epoch, loss, gif_num, side_num):
#         samples = sample(model, scheduler, image_size=self.cfg.dataset.img_size, batch_size = side_num * side_num, channels=self.cfg.dataset.n_channels)
#         _, axarr = plt.subplots(side_num, side_num)
#         for i in range(side_num):
#             for j in range(side_num):
#                 axarr[i,j].imshow(samples[-1][i * side_num + j].reshape(self.cfg.dataset.img_size, self.cfg.dataset.img_size, self.cfg.dataset.n_channels), cmap="gray")
        
#         plt.suptitle(f'{loss=:.4f}', fontsize = 14)
#         plt.tight_layout()
#         plt.savefig(f'../gifs/{experiment_name}__ep{epoch}.png')
#         wandb.log({'image': wandb.Image(f'../gifs/{experiment_name}__ep{epoch}.png'), 'epoch': epoch})

#         # bottleneck is here
#         for i in range(gif_num):
#             self.make_animation(samples, i, experiment_name, epoch)

# # %%
# import numpy as np

# reverse_transform = Compose([
#      Lambda(lambda t: (t + 1) / 2),
#      Lambda(lambda t: t.permute(1, 2, 0)), # CHW to HWC
#      Lambda(lambda t: t * 255.),
#      Lambda(lambda t: t.numpy().astype(np.uint8)),
#      ToPILImage(),
# ])