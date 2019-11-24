#!/usr/bin/env python
# coding: utf-8
import os

os.environ["QT_QPA_PLATFORM"] = "offscreen"
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

import torch
import torch.utils.data
import torchvision

import simutils

torch.manual_seed(12345)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


n_epoch = 2
loaders = simutils.get_loaders(1024, batch_size=2)

train_loss = []
test_loss = []
final_convnet = simutils.ConvNet().to(device)
optimizer = torch.optim.SGD(
    final_convnet.parameters(), lr=0.0001, momentum=0.9, weight_decay=1e-6
)
for i in tqdm(range(n_epoch)):
    final_convnet.train()
    for i_batch, (xy, idx) in enumerate(loaders["train"]):
        optimizer.zero_grad()
        out = final_convnet(xy["chips"].to(device))
        loss = -torch.mean(simutils.get_loglik(xy, out))
        loss.backward()
        optimizer.step()
        train_loss.append(float(loss.detach()))
        
    for i_batch, (xy, idx) in enumerate(loaders["valid"]):
        optimizer.zero_grad()
        out = final_convnet(xy["chips"].to(device))
        loss = -torch.mean(simutils.get_loglik(xy, out))
        loss.backward()
        optimizer.step()
        train_loss.append(float(loss.detach()))

    final_convnet.eval()
    for i_batch, (txy, idx) in enumerate(loaders["test"]):
        tout = final_convnet(txy["chips"].to(device))
        loss = -torch.mean(simutils.get_loglik(txy, tout))
        test_loss.append(float(loss.detach()))

train_loss = np.array(train_loss)
test_loss = np.array(test_loss)

fig = plt.figure()
simutils.plot_loss(train_loss, c="red")
simutils.plot_loss(test_loss, c="blue")
plt.ylabel("Loss")
fig.savefig(f"retrained_loss.png")
