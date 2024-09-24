import torch
import torch.nn as nn


class MeanFunction(nn.Module):
    """
    Base mean function
    """

    def __init__(self, dtype, device):
        super().__init__()
        self.dtype = dtype
        self.device = device

    def __call__(self, X, ind_nan=None):
        raise NotImplementedError("Not Implemented")


class Identity(MeanFunction):
    """
    y = x
    """

    def __init__(self, input_dim=None, dtype=torch.float64, device=None):
        MeanFunction.__init__(self, dtype=dtype, device=device)
        self.input_dim = input_dim

    def __call__(self, X, ind_nan=None):
        """
        @param X: input matrix
        @param ind_nan:
        @return: the same matrix
        """
        return X


class Zero(MeanFunction):
    def __init__(self, output_dim=1, dtype=torch.float64, device=None):
        MeanFunction.__init__(self, dtype=dtype, device=device)
        self.output_dim = output_dim

    def __call__(self, X, ind_nan=None):
        """
        @param X: input
        @param ind_nan:
        @return: zeros similar to x
        """
        return torch.zeros([X.shape[0], self.output_dim]).to(X.dtype).to(self.device)
