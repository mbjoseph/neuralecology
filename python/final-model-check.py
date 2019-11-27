#!/usr/bin/env python
# coding: utf-8
import os

#os.environ["QT_QPA_PLATFORM"] = "offscreen"
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas as pd

import torch
import torch.utils.data
import torchvision

import simutils


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


loaders = simutils.get_loaders(1024, batch_size=1)

final_convnet = simutils.ConvNet().to(device)
final_convnet.load_state_dict(torch.load("out/params/final-conv-hmm.pt"))
final_convnet.eval()

dfs = []

# iterate over test set examples and save estimated transition probabilities
for i_batch, (txy, idx) in enumerate(tqdm(loaders["test"])):
  pred = final_convnet(txy["chips"].to(device))
  loss = -torch.mean(simutils.get_loglik(txy, pred))
  dfs.append(
    pd.DataFrame({
      "pred_gamma_12": pred["Omega"][0, :, 0, 1].detach().cpu().numpy(),
      "pred_gamma_21": pred["Omega"][0, :, 1, 0].detach().cpu().numpy(),
      "chm": txy["chm"].numpy().squeeze(),
      "directory": txy["subdir"][0].split("/")[-1],
      "t": 1 + np.arange(50),
      "loss": float(loss.detach()),
    })
  )
    
df = pd.concat(dfs)
df.to_csv("out/test-set-checks.csv")
