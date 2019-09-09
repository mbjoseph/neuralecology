""" Utility functions for neural dynamic occupancy model training. """

import numpy as np
import torch
from tqdm import tqdm

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.set_default_dtype(torch.float64)


def bbs_nll(xy, model):
    """ Negative log-likelihood for dynamic occupancy model. 
    
    Args
    ----
    xy (tuple): inputs and outputs for the model
    model (torch.nn.Module): a model object to use.
    
    Returns
    -------
    A tuple of:
    - logliks (torch.tensor): log likelihoods for each example in a minibatch
    - out (dict): output from the model, including parameter values
    """
    sp_i, gn_i, fm_i, or_i, l1_i, x_i, x_p_i, y_i = xy
    sp_i = sp_i.to(device)
    gn_i = gn_i.to(device)
    fm_i = fm_i.to(device)
    or_i = or_i.to(device)
    l1_i = l1_i.to(device)
    x_i = x_i.to(device)
    x_p_i = x_p_i.to(device)
    y_i = y_i.to(device)

    # in cases with no surveys:
    # - set the survey covariates to zero (this does not contribute to loss)
    k_i = torch.ones_like(y_i) * 50  # 50 stops
    no_surveys = y_i != y_i
    x_p_i[no_surveys] = 0
    y_i[no_surveys] = 0

    with torch.autograd.detect_anomaly():
        out = model(sp_i, gn_i, fm_i, or_i, l1_i, x_i, x_p_i)
        batch_size = y_i.shape[0]
        num_years = y_i.shape[1]

        lp_y_present = torch.distributions.binomial.Binomial(
            total_count=k_i, logits=out["logit_p"]
        ).log_prob(y_i)
        po = torch.stack(
            (
                torch.exp(lp_y_present),
                (y_i == 0).to(device, dtype=torch.float64),
            ),
            -1,
        )
        # iterate over training examples and replace po with 2x2 identity matrix
        # when no surveys are conducted
        po[no_surveys, :] = torch.ones(2).to(device)
        assert torch.sum(torch.isnan(po)) < 1

        phi_0_i = torch.cat((out["psi0"], 1 - out["psi0"]), -1)
        Omega = torch.stack(
            (
                torch.stack((out["phi"], 1 - out["phi"]), -1),
                torch.stack((out["gamma"], 1 - out["gamma"]), -1),
            ),
            -2,  # stacking along rows (so that rows probs to one)
        )
        assert Omega.shape == (batch_size, num_years - 1, 2, 2)

        c = list()
        # first year: t = 0
        alpha_raw = torch.bmm(
            phi_0_i.unsqueeze(1),
            # batch diag
            torch.diag_embed(po[:, 0, :], dim1=-2, dim2=-1),
        )
        c.append(
            (
                torch.ones(batch_size, 1).to(device)
                / torch.sum(alpha_raw, dim=-1)
            ).squeeze()
        )
        alpha = c[-1].view(-1, 1, 1) * alpha_raw

        # subsequent years: t > 0
        for t in range(num_years - 1):
            tmp = torch.bmm(
                Omega[:, t, :, :],
                # batch diagonal
                torch.diag_embed(po[:, t + 1, :], dim1=-2, dim2=-1),
            )
            alpha_raw = torch.bmm(alpha, tmp)
            c.append(
                (
                    torch.ones(batch_size, 1).to(device)
                    / torch.sum(alpha_raw, dim=-1)
                ).squeeze()
            )
            alpha = c[-1].view(-1, 1, 1) * alpha_raw
        c_stacked = torch.stack(c, -1)
        # log likelihood for each item in the minibatch
        logliks = -torch.sum(torch.log(c_stacked), dim=-1)
    return logliks, out


def fit_epoch(model, loader, training, optimizer=None, pb=True):
    """ Run through a data set for one epoch. 
    
    Args
    ----
    model (torch.nn.Module): model to use
    loader (torch.utils.data.DataLoader): data to use
    training (bool): whether to train the model (True), or just evaluate (False)
    optimizer (torch.optim): an optimizer to update parameter values
    pb (bool): whether to display progress bar while iterating over mini-batches
    
    Returns
    -------
    t_loss (numpy.array): array of minibatch loss values (avg for each batch)
    """
    t_loss = torch.zeros(len(loader))
    t = iter(loader)
    if pb:
        t = tqdm(t)
    for i_batch, xy in enumerate(t):
        logliks, *_ = bbs_nll(xy, model=model)
        loss = -torch.mean(logliks)  # NLL, average over minibatch examples
        if training:
            with torch.autograd.detect_anomaly():
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        t_loss[i_batch] = loss
    t_loss = t_loss.cpu().detach().numpy()
    return t_loss
