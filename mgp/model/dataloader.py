import torch

# import numpy as np
# torch.multiprocessing.set_start_method('spawn')


class CDataSet(torch.utils.data.Dataset):
    def __init__(self, X, y, device="cpu", dtype=torch.float64):
        self.X = X
        self.y = y
        self.device = device
        self.dtype = dtype

    def __len__(self):
        return len(self.y)

    def __getitem__(self, item):
        X = self.X[item, :]
        y = self.y[item, :]
        x = torch.from_numpy(X).to(self.dtype)
        y = torch.from_numpy(y).to(self.dtype)
        return x, y[0]
