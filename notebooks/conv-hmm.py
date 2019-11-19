#!/usr/bin/env python
# coding: utf-8
import os

os.environ["QT_QPA_PLATFORM"] = "offscreen"
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.utils.data
import torchvision
import torch.nn.functional as F
import torch.nn as nn

import simutils

torch.manual_seed(12345)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Model definitions


class Net(torch.nn.Module):
    """ Definition for the 'oracle' model
    
    This model receives the "correct" covariate (canopy height) 
    which is used in the actual data generation process.
    """

    def __init__(self):
        super(Net, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(1, 64), nn.LeakyReLU(), nn.Linear(64, 4)
        )
        self.gamma_pars = torch.nn.Parameter(torch.randn(2, 2) * 0.1)
        self.loc_pars = torch.nn.Parameter(torch.randn(2) * 0.1)
        self.conc_pars = torch.nn.Parameter(torch.randn(2) * 0.1)

    def forward(self, x):
        # x should be (batch_size, nt, 1)
        nt = x.shape[1]
        x = self.fc(x)
        Omega = F.softmax(x.view(-1, nt, 2, 2), dim=-1)
        return {
            "Omega": Omega,
            "gamma_pars": torch.exp(self.gamma_pars),
            "conc_pars": torch.exp(self.conc_pars),
            "loc_pars": self.loc_pars,  # VonMises location is unconstrained
        }


class ConvNet(torch.nn.Module):
    """ Definition for a convolutional movement model
    
    This model takes "chips" as input, where each chip
    is a tile from an RGB image centered on the animal's
    measured location. 
    
    This convolutional neural network maps image chips to 
    state transition probability matrices:
    
    Omega = f(image chip), 
    
    where Omega is a transition matrix and f is the convnet.
    """

    def __init__(self):
        super(ConvNet, self).__init__()
        # scalar parameters
        self.gamma_pars = torch.nn.Parameter(torch.randn(2, 2) * 0.1)
        self.loc_pars = torch.nn.Parameter(torch.randn(2) * 0.1)
        self.conc_pars = torch.nn.Parameter(torch.randn(2) * 0.1)

        self.conv1 = nn.Sequential(
            nn.Conv2d(4, 16, kernel_size=9, stride=3), nn.LeakyReLU(),
        )
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2)

        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=5), nn.LeakyReLU(),
        )
        self.pool2 = nn.MaxPool2d(kernel_size=3, stride=2)

        self.conv3 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3), nn.LeakyReLU(),
        )
        self.pool3 = nn.MaxPool2d(kernel_size=3, stride=2)

        self.fc = nn.Sequential(
            nn.Dropout(),
            nn.Linear(64 * 2 * 2, 128),
            nn.LeakyReLU(),
            nn.Dropout(),
            nn.Linear(128, 64),
            nn.LeakyReLU(),
            nn.Linear(64, 4),
        )

    def forward(self, x):
        batch_size = x.shape[0]
        nt = x.shape[1]
        batch_times_nt = batch_size * nt
        # x is an RGB chip of shape (batch, nt, channels, width, height)
        # reshape to (batch * nt, channels, width, height)
        # as advised https://github.com/pytorch/pytorch/issues/21688
        x = x.view(batch_times_nt, 4, 128, 128)
        x = self.conv1(x)
        # print(f"After conv1: {x.shape}")
        x = self.pool1(x)
        # print(f"After pool1: {x.shape}")
        x = self.conv2(x)
        # print(f"After conv2: {x.shape}")
        x = self.pool2(x)
        # print(f"After pool2: {x.shape}")
        x = self.conv3(x)
        # print(f"After conv3: {x.shape}")
        x = self.pool3(x)
        x = x.view(batch_times_nt, -1)  # flatten
        x = self.fc(x)

        # reshape to (batch, nt, 2, 2) for transition matrices
        Omega = F.softmax(x.view(-1, nt, 2, 2), dim=-1)
        return {
            "Omega": Omega,
            "gamma_pars": torch.exp(self.gamma_pars),
            "conc_pars": torch.exp(self.conc_pars),
            "loc_pars": self.loc_pars,  # VonMises location is unconstrained
        }


class PtNet(torch.nn.Module):
    """ Pointwise RGB extraction network
    
    This model only uses the RGB reflectance data at 
    the point location where the animal was located.
    """

    def __init__(self):
        super(PtNet, self).__init__()
        # affine operations: y = Wx + b
        self.fc1 = torch.nn.Linear(3, 64)
        self.fc2 = torch.nn.Linear(64, 4)
        self.gamma_pars = torch.nn.Parameter(torch.randn(2, 2) * 0.1)
        self.loc_pars = torch.nn.Parameter(torch.randn(2) * 0.1)
        self.conc_pars = torch.nn.Parameter(torch.randn(2) * 0.1)
        self.softmax = torch.nn.Softmax(dim=-1)

    def forward(self, x):
        # x should be (batch_size, nt, 1)
        nt = x.shape[1]
        output = self.fc2(torch.relu(self.fc1(x)))
        Omega = self.softmax(output.view(-1, nt, 2, 2))
        return {
            "Omega": Omega,
            "gamma_pars": torch.exp(self.gamma_pars),
            "conc_pars": torch.exp(self.conc_pars),
            "loc_pars": self.loc_pars,  # VonMises location is unconstrained
        }


# # Train the models
n_epoch = 100  # Good results with 100 epochs, no weight decay
train_data_sizes = [2 ** i for i in range(6, 11)]  # 2**8=256, 2**11=2048
print(train_data_sizes)

for n in train_data_sizes:
    print(f"Training with sample size {n}")
    loaders = simutils.get_loaders(n, batch_size=2)
    convnet = simutils.fit(ConvNet, "chips", loaders, n_epoch=n_epoch)
    oracle = simutils.fit(Net, "chm", loaders, n_epoch=n_epoch)
    ptnet = simutils.fit(PtNet, "rgb_pt", loaders, n_epoch=n_epoch)

    fig = plt.figure()
    simutils.plot_loss(oracle.get("train_loss"))
    simutils.plot_loss(convnet.get("train_loss"), c="red")
    simutils.plot_loss(ptnet.get("train_loss"), c="green")
    plt.ylabel("Training loss")
    fig.savefig(f"train_loss_n{n}.png")

    fig = plt.figure()
    simutils.plot_loss(oracle.get("valid_loss"))
    simutils.plot_loss(convnet.get("valid_loss"), c="red")
    simutils.plot_loss(ptnet.get("valid_loss"), c="green")
    plt.ylabel("Validation loss")
    fig.savefig(f"valid_loss_n{n}.png")

    del oracle
    del convnet
    del ptnet
