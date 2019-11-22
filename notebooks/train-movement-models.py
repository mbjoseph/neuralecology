#!/usr/bin/env python
# coding: utf-8
import os

os.environ["QT_QPA_PLATFORM"] = "offscreen"
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.utils.data
import torchvision

import simutils

torch.manual_seed(12345)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False



# Train the models
n_epoch = 100
train_data_sizes = [2 ** i for i in range(4, 11)]
print(train_data_sizes)

for n in train_data_sizes:
    print(f"Training with sample size {n}")
    loaders = simutils.get_loaders(n, batch_size=2)
    convnet = simutils.fit(simutils.ConvNet, "chips", loaders, n_epoch=n_epoch)

    fig = plt.figure()
    simutils.plot_loss(convnet.get("train_loss"), c="red")
    plt.ylabel("Training loss")
    fig.savefig(f"train_loss_n{n}.png")

    fig = plt.figure()
    simutils.plot_loss(convnet.get("valid_loss"), c="red")
    plt.ylabel("Validation loss")
    fig.savefig(f"valid_loss_n{n}.png")

    del convnet
