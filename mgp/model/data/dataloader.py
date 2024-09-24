import os.path
import random
import torch
import numpy as np
import abc
import torch.utils.data as data


class baseData(data.Dataset):
    def __init__(self):
        super(baseData, self).__init__()

    @abc.abstractmethod
    def create(self, opt, x, y):
        raise NotImplementedError("Subclass should be implemented.")

    @abc.abstractmethod
    def __len__(self):
        raise NotImplementedError("Subclass should be implemented.")

    @abc.abstractmethod
    def name(self):
        raise NotImplementedError("Subclass should be implemented.")


class dataloader(baseData):
    def name(self):
        return "DataLoader"

    def create(self, X, y, ind_nan, ind_nan_target, device=None, dtype=torch.float64):
        """
        options of the data set
        :param opt:
        :return:
        """

        self.X = X
        self.y = None
        if y is not None:
            self.y = y
        self.ind_nan = ind_nan
        self.device = device
        self.dtype = dtype
        if ind_nan_target.ndim == 1:
            ind_nan_target = ind_nan_target.reshape(-1, 1)
        self.ind_nan_target = ind_nan_target

    def __getitem__(self, item):
        """
        randomly select one image form the pool
        :param item:
        :return:
        """
        X = self.X[:, item, :]

        ind_nan = self.ind_nan[item, :]
        x = torch.from_numpy(X).to(self.dtype)  # .to(self.device)

        y = self.y[item, :]
        y = torch.from_numpy(y).to(self.dtype)  # .to(self.device)
        # if y.ndim == 1:
        #   y = y.unsqueeze(-1)
        ind_nan_target = self.ind_nan_target[item, :]
        return x, y, ind_nan, ind_nan_target

    def __len__(self):
        return self.y.shape[0]
