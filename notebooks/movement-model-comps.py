#!/usr/bin/env python
# coding: utf-8
import os
import glob

os.environ["QT_QPA_PLATFORM"] = "offscreen"
import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.utils.data
import torchvision
import torch.nn.functional as F
import torch.nn as nn

import simutils

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# Compute validation log likelihood for neural hierarchical models
train_data_sizes = [2 ** i for i in range(4, 11)]  # 2**11=2048
print(train_data_sizes)


def get_loss(model, xy, input_name):
    out = model(xy[input_name].to(device))
    loss = -torch.sum(simutils.get_loglik(xy, out))
    return float(loss.detach())


def bestcase_Omega(chm, coefs):
    """ Transition probability matrix for Omega """
    # chm is shaped (batch_size, nt, 1)
    # coefs is shaped (2, 2) where the
    # first row is intercept and slope for state transition 1 -> 2
    # second row is intercept and slope for state transition 2 -> 1
    t1to2 = torch.sigmoid(coefs[0, 0] + chm * coefs[0, 1])
    t2to1 = torch.sigmoid(coefs[1, 0] + chm * coefs[1, 1])
    t1to1 = 1 - t1to2
    t2to2 = 1 - t2to1
    Omega_row1 = torch.cat((t1to1, t1to2), -1)
    Omega_row2 = torch.cat((t2to1, t2to2), -1)
    Omega = torch.stack((Omega_row1, Omega_row2), -2)
    assert Omega.shape == (chm.shape[0], chm.shape[1], 2, 2)
    return Omega


def pt_Omega(rgb_pt, coefs):
    """ Transition probability matrix for Omega """
    # rgb_pt is shaped (batch_size, nt, 3)
    # coefs is shaped (2, 4) where the
    # first row is intercept and 3 slopes for state transition 1 -> 2
    # second row is intercept and 3 slopes for state transition 2 -> 1
    t1to2 = torch.sigmoid(
        coefs[0, 0] + torch.sum(rgb_pt * coefs[0, 1:].view(1, 1, -1), -1)
    )
    t2to1 = torch.sigmoid(
        coefs[1, 0] + torch.sum(rgb_pt * coefs[1, 1:].view(1, 1, -1), -1)
    )
    t1to1 = 1 - t1to2
    t2to2 = 1 - t2to1
    Omega_row1 = torch.stack((t1to1, t1to2), -1)
    Omega_row2 = torch.stack((t2to1, t2to2), -1)
    Omega = torch.stack((Omega_row1, Omega_row2), -2)
    assert Omega.shape == (rgb_pt.shape[0], rgb_pt.shape[1], 2, 2)
    return Omega


for n in train_data_sizes:
    loaders = simutils.get_loaders(n, batch_size=8, shuffle_validation=False)

    # ConvNet
    convnet = simutils.ConvNet()
    convnet.load_state_dict(torch.load(f"../out/params/ConvNet_{n}_params.pt"))
    convnet.to(device)
    convnet.eval()

    # baseline bestcase model
    bestcase_wt = pd.read_csv(f"../out/params/bestcase_{n}.csv")
    bestcase_gamma_pars = torch.tensor(
        bestcase_wt[["gamma_shape", "gamma_rate"]].values, dtype=torch.float
    )
    bestcase_loc_pars = torch.tensor(
        np.squeeze(bestcase_wt[["vm_mean"]].values), dtype=torch.float
    )
    bestcase_conc_pars = torch.tensor(
        np.squeeze(bestcase_wt[["vm_concentration"]].values), dtype=torch.float
    )
    bestcase_coefs = torch.tensor(
        bestcase_wt[["transition_intercept", "z"]].values, dtype=torch.float
    )

    # baseline pt model
    pt_wt = pd.read_csv(f"../out/params/ptextract_{n}.csv")
    pt_gamma_pars = torch.tensor(
        pt_wt[["gamma_shape", "gamma_rate"]].values, dtype=torch.float
    )
    pt_loc_pars = torch.tensor(
        np.squeeze(pt_wt[["vm_mean"]].values), dtype=torch.float
    )
    pt_conc_pars = torch.tensor(
        np.squeeze(pt_wt[["vm_concentration"]].values), dtype=torch.float
    )
    pt_coefs = torch.tensor(
        pt_wt[
            [
                "transition_intercept",
                "rgb_mosaic.1",
                "rgb_mosaic.2",
                "rgb_mosaic.3",
            ]
        ].values,
        dtype=torch.float,
    )

    conv_loss = []
    bestcase_loss = []
    baseline_pt_loss = []
    batch_len = []
    print(f"Computing validation loss for n={n}")
    for i_batch, (xy, idx) in enumerate(tqdm(loaders["valid"])):
        conv_loss.append(get_loss(convnet, xy, "chips"))

        # compute losses for baseline models
        step_size = xy["step_size"]
        turn_angle = xy["turn_angle"]
        bestcase_loglik = simutils.forward_algorithm(
            step_size,
            turn_angle,
            bestcase_Omega(xy["chm"], bestcase_coefs),
            bestcase_gamma_pars,
            bestcase_loc_pars,
            bestcase_conc_pars,
        )
        bestcase_loss.append(float(-torch.sum(bestcase_loglik.detach())))
        pt_loglik = simutils.forward_algorithm(
            step_size,
            turn_angle,
            pt_Omega(xy["rgb_pt"], pt_coefs),
            pt_gamma_pars,
            pt_loc_pars,
            pt_conc_pars,
        )
        baseline_pt_loss.append(float(-torch.sum(pt_loglik.detach())))
        batch_len.append(len(idx))

    loss_df = pd.DataFrame(
        {
            "n": n,
            "conv_loss": conv_loss,
            "bestcase_loss": bestcase_loss,
            "baseline_pt_loss": baseline_pt_loss,
            "batch_size": batch_len,
        }
    )
    print(loss_df)
    loss_df.to_csv(f"../out/params/loss_df_{n}.csv")
