# Datasets and Dataloaders

import torch
import torchvision
import torchvision.transforms as transforms

# Change image to pre-setted form
transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ]
)

# Load prepared dataset
trainset = torchvision.datasets.CIFAR10(
    root="./data",
    train=True,
    download=True,
    transform=transform,
)

# Load data loader with dataset
trainloader = torch.utils.data.DataLoader(
    trainset,
    batch_size=4,
    shuffle=True,
    num_workers=2,
)


import matplotlib.pyplot as plt
import numpy as np
from multiprocessing import freeze_support

classes = (
    "plane",
    "car",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
)


def imshow(img):
    # unnormalize
    # mean = 0.5, std = 0.5
    img = img / 2 + 0.5
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


def load_dataset():
    # get some random training images
    dataiter = iter(trainloader)
    images, labels = dataiter.next()

    # show images
    imshow(torchvision.utils.make_grid(images))
    # print labels
    print(" ".join("%5s" % classes[labels[j]] for j in range(4)))


# https://stackoverflow.com/questions/50701690/pytorch-tutorial-error-training-a-classifier
if __name__ == "__main__":
    freeze_support()
    load_dataset()
