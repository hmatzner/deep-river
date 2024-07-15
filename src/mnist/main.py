import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import make_grid

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt


class Preprocessing:
    transform = transforms.ToTensor()

    train_data = datasets.MNIST(
        root='cnn_data',
        train=True,
        download=True,
        transform=transform
    )

    test_data = datasets.MNIST(
        root='cnn_data',
        train=False,
        download=True,
        transform=transform
    )

    # print(test_data)


class ConvolutionalNetwork(nn.Module):
    pass


def main():
    Preprocessing()
    ConvolutionalNetwork()


if __name__ == '__main__':
    main()
