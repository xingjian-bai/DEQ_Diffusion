# %%
from torchvision.transforms import Compose, ToTensor, Lambda, ToPILImage, CenterCrop, Resize
from datasets import load_dataset
from torchvision import transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import torch
# load dataset from the hub
# dataset = load_dataset("fashion_mnist")
# %%
# torch.save(dataset, "../data/fashion_mnist.pkl")
# print('started loading dataset')

# IMPORTANT: current position is in src/output/time/logs
dataset = torch.load("../../../../data/fashion_mnist.pkl")
IMAGE_SIZE = 28
CHANNELS = 1
BATCH_SIZE = 128
# %%
transform = Compose([
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Lambda(lambda t: (t * 2) - 1)
])
def transforms(examples):
   examples["pixel_values"] = [transform(image.convert("L")) for image in examples["image"]]
   del examples["image"]
   return examples
# print('finished loading dataset')
transformed_dataset = dataset.with_transform(transforms).remove_columns("label")
# %%
# # create dataloader
dataloader = DataLoader(transformed_dataset["train"], batch_size=BATCH_SIZE, shuffle=True)
test_dataloader = DataLoader(transformed_dataset["test"], batch_size=BATCH_SIZE, shuffle=True)
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

def make_animation(samples, scheduler, index, name_prefix):
    fig = plt.figure()
    ims = []
    for i in range(scheduler.timesteps):
        im = plt.imshow(samples[i][index].reshape(IMAGE_SIZE, IMAGE_SIZE, CHANNELS), cmap="gray", animated=True)
        ims.append([im])
    animate = animation.ArtistAnimation(fig, ims, interval=50, blit=True, repeat_delay=1000)
    
    id = 0
    while os.path.exists(f'../gifs/{name_prefix}_{id}.gif'):
        id += 1
    animate.save(f'../gifs/{name_prefix}_{id}.gif')

def visualize (model, scheduler, gif_num = 1, side_num = 4, name_prefix = "unnamed"):
    samples = sample(model, scheduler, image_size=IMAGE_SIZE, batch_size = side_num * side_num, channels=CHANNELS)
    
    f, axarr = plt.subplots(side_num, side_num)
    for i in range(side_num):
        for j in range(side_num):
            axarr[i,j].imshow(samples[-1][i * side_num + j].reshape(IMAGE_SIZE, IMAGE_SIZE, CHANNELS), cmap="gray")
    
    id = 0
    while os.path.exists(f'../gifs/{name_prefix}_{id}.gif'):
        id += 1
    plt.savefig(f'../gifs/{name_prefix}_{id}.png')

    # bottleneck is here
    for i in range(gif_num):
        make_animation(samples, scheduler, i, name_prefix = name_prefix)
# %%
from dataset import CleanDataset

class FasionMNISTDataset(CleanDataset):
    def DataLoader(self):
        return dataloader
    
    # def Dataset(self):
    #     return transformed_dataset
    
    def TestDataloader(self):
        return test_dataloader
    
    def visualize (self, gif_num = 1, side_num = 4, name_prefix = "unamed"):
        return visualize (self, gif_num = gif_num, side_num = side_num, name_prefix = name_prefix)

# # %%
# import numpy as np

# reverse_transform = Compose([
#      Lambda(lambda t: (t + 1) / 2),
#      Lambda(lambda t: t.permute(1, 2, 0)), # CHW to HWC
#      Lambda(lambda t: t * 255.),
#      Lambda(lambda t: t.numpy().astype(np.uint8)),
#      ToPILImage(),
# ])