import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

# Load all of the BBS data into a DataFrame
bbs = pd.merge(
    pd.read_csv("data/cleaned/bbs.csv"),
    pd.read_csv("data/cleaned/clean_routes.csv"),
    how="left",
    on="route_id",
)

# Create dictionaries that map categories to integers
categorical_covariates = ["english", "genus", "family", "order", "L1_KEY"]
cat_ix = {}
for cat in categorical_covariates:
    cat_dict = {}
    category_series = bbs[cat].unique()
    for i, item in enumerate(category_series):
        cat_dict[item] = i
    cat_ix[cat] = cat_dict


class BBSData(Dataset):
    """North American Breeding Bird Survey data."""

    def __init__(self, df):
        """
        Args:
            df (pandas.DataFrame): data frame with bbs data.
        """
        self.df = df
        self.cat_ix = cat_ix
        self.bbs_y = self.get_cont("^[0-9]{4}$", df)
        self.x_p = torch.stack(
            (
                self.get_cont("^StartTemp_", df),
                self.get_cont("^StartWind_", df),
                self.get_cont("^EndTemp_", df),
                self.get_cont("^EndWind_", df),
                self.get_cont("^StartSky_", df),
                self.get_cont("^EndSky_", df),
                self.get_cont("^duration_", df),
            ),
            -1,
        )
        self.bbs_species = self.get_cat("english", df)
        self.bbs_genus = self.get_cat("genus", df)
        self.bbs_family = self.get_cat("family", df)
        self.bbs_order = self.get_cat("order", df)
        self.bbs_l1 = self.get_cat("L1_KEY", df)
        # x is a covariate vector, e.g., PC1, PC2, ...
        self.bbs_x = torch.tensor(
            df.filter(regex="^PC|^c_", axis=1).values, dtype=torch.float64
        )

    def __len__(self):
        return len(self.df.index)

    def get_cont(self, regex, df):
        """ Extract continuous valued data from columns matching regex. """
        res = torch.tensor(
            df.filter(regex=regex, axis=1).values, dtype=torch.float64
        )
        return res

    def get_cat(self, name, df):
        """ Find an integer index for a particular category. """
        res = [self.cat_ix[name][i[0]] for i in self.df[[name]].values]
        res = np.array(res, dtype=np.long)
        return torch.tensor(res, dtype=torch.long)

    def __getitem__(self, idx):
        """ Get an item from the data. 
        
        This is one training example: a route X species time series with feats
        """
        y = self.bbs_y[idx, :].squeeze(0)
        x_p = self.x_p[idx, :, :].squeeze(0)
        x = self.bbs_x[idx, :].squeeze(0)
        species = self.bbs_species[idx]
        genus = self.bbs_genus[idx]
        family = self.bbs_family[idx]
        order = self.bbs_order[idx]
        l1 = self.bbs_l1[idx]
        return species, genus, family, order, l1, x, x_p, y
