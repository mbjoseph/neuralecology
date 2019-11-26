""" Utility functions for convolutional neural movement model """
import os
import glob
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import multiprocessing
import pandas as pd
import PIL
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import pyro.distributions

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

""" Data munging """


class TrajectoryDataset(torch.utils.data.Dataset):
    """Movement trajectory dataset."""

    def __init__(self, src_dir, train=False, nmax=None):
        """
        Args:
            src_dir (string): Directory with image chips and trajectory data.
            nmax (int): Optional maximum number of trajectories to use.
          
        Returns: 
            A new instance of a trajectory dataset.
        """
        self.src_dir = src_dir
        self.subdirs = sorted(glob.glob(os.path.join(src_dir, "*")))
        self.chip_nxy = 128
        self.train = train
        if nmax:
            self.subdirs = sorted(self.subdirs)[:nmax]

    def __len__(self):
        return len(self.subdirs)

    def train_transforms(self):
        trans = torchvision.transforms.Compose(
            [
                torchvision.transforms.RandomHorizontalFlip(),
                torchvision.transforms.RandomVerticalFlip(),
            ]
        )
        return trans

    def common_transforms(self):
        trans = torchvision.transforms.Compose(
            [torchvision.transforms.ToTensor(),]
        )
        return trans

    def _load_chip(self, path):
        chip = PIL.Image.open(path)

        if self.train:
            self.train_transforms()(chip)
        chip = self.common_transforms()(chip)

        # binary indicator for center point (to indicate position of animal)
        # as a way to retain translation invariance, i.e., an
        # attention mask concatenated to the input image as another channel
        centerpt = torch.zeros(self.chip_nxy, self.chip_nxy)
        centerpt[
            (self.chip_nxy - 1) : self.chip_nxy,
            (self.chip_nxy - 1) : self.chip_nxy,
        ] = 1
        chip = torch.cat((chip, centerpt.view(1, self.chip_nxy, self.chip_nxy)))

        return chip

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        subdir = self.subdirs[idx]

        chips = glob.glob(os.path.join(subdir, "chip*.tiff"))
        chip_tensors = torch.stack(
            [self._load_chip(c) for c in sorted(chips)], dim=0
        )

        assert chip_tensors.shape == (
            50,
            4,
            self.chip_nxy,
            self.chip_nxy,
        ), f"{subdir} has wrong shape"

        trajectory = pd.read_csv(os.path.join(subdir, "coords.csv"))
        turn_angle = torch.tensor(
            trajectory.turn_angle.values, dtype=torch.float
        )
        step_size = torch.tensor(trajectory.step_size.values, dtype=torch.float)
        chm = torch.tensor(trajectory.z.values, dtype=torch.float)
        rgb_pt = torch.tensor(
            trajectory[
                [col for col in trajectory if col.startswith("rgb_")]
            ].values,
            dtype=torch.float,
        )

        sample = {
            "chips": chip_tensors,
            "turn_angle": turn_angle,
            "step_size": step_size,
            "stationary_p1": torch.tensor(
                trajectory.stationary_p1.values, dtype=torch.float
            ),
            "stationary_p2": torch.tensor(
                trajectory.stationary_p2.values, dtype=torch.float
            ),
            "chm": chm.unsqueeze(-1),
            "rgb_pt": rgb_pt / 255,
            "subdir": subdir
        }
        return sample, idx


def get_loaders(nmax, batch_size):
    """ Get DataLoader objects of train/valid sets.
    
    Args: 
        nmax (int): number of training examples to use
    """
    datasets = {
        "train": TrajectoryDataset(
            "../out/trajectories/train", train=True, nmax=nmax
        ),
        "validation": TrajectoryDataset("../out/trajectories/validation"),
        "test": TrajectoryDataset("../out/trajectories/test")
    }
    n_cpu = multiprocessing.cpu_count()
    dataloaders = {
        "train": torch.utils.data.DataLoader(
            datasets["train"],
            batch_size=batch_size,
            shuffle=True,
            num_workers=n_cpu,
        ),
        "valid": torch.utils.data.DataLoader(
            datasets["validation"],
            batch_size=batch_size,
            shuffle=True,
            num_workers=n_cpu,
        ),
        "test": torch.utils.data.DataLoader(
            datasets["test"],
            batch_size=batch_size,
            shuffle=True,
            num_workers=n_cpu,
        ),
    }
    return dataloaders


""" Modeling utilities """


def fit(Model, input_name, loaders, n_epoch=1):
    """ Create and fit a model.
    
    Args: 
        Model (class): class definition for the model
        input_name (string): which input to use, i.e., the name of x
        loaders (dict): dictionary of training and validation dataloaders
        n_epoch (int): number of training epochs
    
    Returns: 
        A dictionary containing the trained model and loss arrays.
    """
    train_loss = []
    valid_loss = []
    model = Model().to(device)
    optimizer = torch.optim.SGD(
        model.parameters(), lr=0.0002, momentum=0.9, weight_decay=1e-5
    )
    for i in tqdm(range(n_epoch)):
        model.train()
        for i_batch, (xy, idx) in enumerate(loaders["train"]):
            optimizer.zero_grad()
            out = model(xy[input_name].to(device))
            loss = -torch.mean(get_loglik(xy, out))
            loss.backward()
            optimizer.step()
            train_loss.append(float(loss.detach()))

        model.eval()
        for i_batch, (vxy, idx) in enumerate(loaders["valid"]):
            if i_batch < 5:  # evaluate on a random subset
                vout = model(vxy[input_name].to(device))
                loss = -torch.mean(get_loglik(vxy, vout))
                valid_loss.append(float(loss.detach()))
    train_loss = np.array(train_loss)
    valid_loss = np.array(valid_loss)
    n_training_examples = len(loaders["train"].dataset)
    save_model(model, n_training_examples, train_loss, valid_loss)
    return {"model": model, "train_loss": train_loss, "valid_loss": valid_loss}


def save_model(model, n, train_loss, valid_loss):
    """ Save parameters & losses of a model object 
    
    Saves to a file, e.g.,:
    ../out/params/ModelClass_samplesize_params.pt
    
    Args: 
        model (nn.Module): model with params to save
        n (int): training set size
        train_loss (numpy.array): training loss values
        valid_loss (numpy.array): validation loss values
    """
    par_dir = os.path.join("..", "out", "params")
    os.makedirs(par_dir, exist_ok=True)
    model_class = type(model).__name__
    output_path = os.path.join(par_dir, f"{model_class}_{n}_params.pt")
    torch.save(model.state_dict(), output_path)

    loss_df = pd.DataFrame(
        {
            "model": model_class,
            "n_train": n,
            "loss": np.concatenate((train_loss, valid_loss)),
            "group": np.array(
                (["train"] * len(train_loss) + ["valid"] * len(valid_loss))
            ),
        }
    )
    loss_df.to_csv(os.path.join(par_dir, f"{model_class}_{n}_loss.csv"))


def get_stationary_probs(Omega):
    """ Compute stationary probabilities of a transition matrix 
    
    This returns a detached array containing stationary probability estimates, 
    which is useful for downstream visualization.
    
    Args: 
      Omega (tensor): a tensor containing transition probability matrices of 
        shape (batch_size, timesteps, n_state, n_state), where the last two 
        dimensions correspond to state transition matrices.
    
    Returns:
      A numpy array containing stationary probabilities of each state, of 
      shape (batch_size, timesteps, n_state)
    """
    g12 = Omega[:, :, 0, 1]
    g21 = Omega[:, :, 1, 0]
    trans_probs = torch.stack((g12, g21), -1)
    p = trans_probs / torch.sum(trans_probs, -1, keepdim=True)
    return p.detach().cpu().numpy()


def get_loglik(xy, pars):
    """ Calculate log likelihood with (scaled) forward algorithm.
    
    Args:
      xy (dict): a batch of data from a dataloader
      pars (dict): parameter output from a model
      
    Returns: 
      A tensor with log-likelihood values for each trajectory in the batch.
    """
    step_size = xy["step_size"].to(device)
    turn_angle = xy["turn_angle"].to(device)

    Omega = pars["Omega"]
    gamma_pars = pars["gamma_pars"]
    loc_pars = pars["loc_pars"]
    conc_pars = pars["conc_pars"]

    loglik = forward_algorithm(
        step_size, turn_angle, Omega, gamma_pars, loc_pars, conc_pars
    )
    return loglik


def forward_algorithm(
    step_size, turn_angle, Omega, gamma_pars, loc_pars, conc_pars
):
    """Scaled forward algorithm for sequence log likelihood. """
    batch_size = step_size.shape[0]
    nt = step_size.shape[1]

    # Generate step size distributions
    step_d1 = torch.distributions.Gamma(gamma_pars[0, 0], gamma_pars[0, 1])
    step_d2 = torch.distributions.Gamma(gamma_pars[1, 0], gamma_pars[1, 1])
    angle_d1 = pyro.distributions.VonMises(loc_pars[0], conc_pars[0])
    angle_d2 = pyro.distributions.VonMises(loc_pars[1], conc_pars[1])

    # compute emission probabilities
    step_p1 = step_d1.log_prob(step_size)
    step_p2 = step_d2.log_prob(step_size)
    angle_p1 = angle_d1.log_prob(turn_angle)
    angle_p2 = angle_d2.log_prob(turn_angle)

    # po gives the probability of each observation, conditional on the state.
    # first observation only contains step size (turn angle needs two vectors)
    po_first = torch.stack(
        (torch.exp(step_p1[:, 0]), torch.exp(step_p2[:, 0])), -1
    ).unsqueeze(1)
    assert po_first.shape == (batch_size, 1, 2)

    # subsequent time steps contain step sizes AND turn angles
    po_subsequent = torch.stack(
        (
            torch.exp(step_p1[:, 1:] + angle_p1[:, 1:]),
            torch.exp(step_p2[:, 1:] + angle_p2[:, 1:]),
        ),
        -1,
    )
    assert po_subsequent.shape == (batch_size, nt - 1, 2)

    po = torch.cat((po_first, po_subsequent), dim=1)  # stack in time dimension
    assert po.shape == (batch_size, nt, 2)

    # initial state probabilities are stationary distribution probs
    initial_omega_21 = Omega[
        :, 0, 1, 0
    ]  # first times, second row, first column
    initial_omega_12 = Omega[:, 0, 0, 1]  # first time, first row, second column
    initial_omega = torch.stack((initial_omega_21, initial_omega_12), -1)
    delta = initial_omega / torch.sum(initial_omega)  # stationary distribution
    assert delta.shape == (batch_size, 2)
    assert Omega.shape == (batch_size, nt, 2, 2)

    c = list()
    alpha_raw = torch.bmm(
        delta.unsqueeze(1),
        # batch diagonal
        torch.diag_embed(po[:, 0, :], dim1=-2, dim2=-1),
    )
    c.append((torch.pow(torch.sum(alpha_raw, dim=-1), exponent=-1)).squeeze())
    alpha = c[-1].view(-1, 1, 1) * alpha_raw

    for t in range(nt - 1):
        alpha_raw = torch.bmm(
            alpha,
            torch.bmm(
                Omega[:, t + 1, :, :],
                # batch diagonal
                torch.diag_embed(po[:, t + 1, :], dim1=-2, dim2=-1),
            ),
        )
        c.append(
            (torch.pow(torch.sum(alpha_raw, dim=-1), exponent=-1)).squeeze()
        )
        alpha = c[-1].view(-1, 1, 1) * alpha_raw
    c_stacked = torch.stack(c, -1)
    return -torch.sum(torch.log(c_stacked), dim=-1)


""" Visualization utilities"""


def plot_loss(x, c="b", alpha=1):
    """ Plot the loss over time on a log scale. 
  
    Args:
      x (list): List containing loss values
      c (str): color to use for the line plot
      alpha (float): transparency for the line plot
    """
    plt.plot(np.arange(len(x)) / len(x), x, alpha=alpha, c=c)
    plt.yscale("log")


def plot_stationary_probs(out, xy, which_prob=1):
    """ Plot estimated vs. true stationary probabilities. 
    
    Args:
        out (dict): output from a model, containing parameter estimates
        xy (dict): a batch of data, containing true values
        which_prob (int): either 1 or 2, to indicate which state to use. This
          argument is useful because the model might assign either state 
          (e.g., in transit or foraging) to state 1 or 2.
    """
    plt.plot(
        np.array([0, 1]), np.array([0, 1]), c="k", linestyle="--", alpha=0.2
    )

    x = xy[f"stationary_p{which_prob}"].view(-1).numpy()
    x_jittered = x + np.random.normal(size=x.shape) * 0.001
    plt.scatter(
        x_jittered,
        # pull first row of trans. matrix for stationary prob plots
        get_stationary_probs(out["Omega"])[:, :, 0].flatten(),
        c="r",
    )
    plt.xlabel("True stationary probability")
    plt.ylabel("Estimated stationary probability")


class ConvNet(nn.Module):
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
        self.gamma_pars = nn.Parameter(torch.randn(2, 2) * 0.1)
        self.loc_pars = nn.Parameter(torch.randn(2) * 0.1)
        self.conc_pars = nn.Parameter(torch.randn(2) * 0.1)

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
            nn.Dropout(),
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
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.conv3(x)
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
