# Credit to GPFlow.

import torch
import torch.nn as nn
import numpy as np


class Gaussian(nn.Module):
    """"""

    def __init__(self, variance=1.0, dtype=torch.float64, device=None, **kwargs):
        """"""
        super().__init__()
        self.dtype = dtype
        self.device = device

        self.register_parameter(
            "log_variance",
            nn.Parameter(
                torch.tensor(np.log(variance), dtype=self.dtype, device=self.device)
            ),
        )

        self.rmse_metric = torch.nn.MSELoss()

        self.nll_metric = torch.mean

    @property
    def metrics(self):
        return [self.rmse_metric, self.nll_metric]

    def update_metrics(
        self,
        y,
        mean_pred,
        std_pred,
        ind_nan=None,
        layer_missing_index=None,
        max_use=None,
    ):
        """
        @param y: output
        @param mean_pred: mean prediction results
        @param std_pred: standard deviation prediction
        @param ind_nan: index of missing values
        @param layer_missing_index: missing index in each layer
        @return: updated metrics
        """

        def compute_nn(y_t, mean_t, std_t):
            S = mean_t.shape[0]
            normal = torch.distributions.Normal(loc=mean_t, scale=std_t)
            logpdf = normal.log_prob(y_t)
            nll = torch.logsumexp(logpdf, 0) - np.log(S)
            return -nll.mean()

        # tiled_y = y.unsqueeze(0).tile([mean_pred[0].shape[0],1,1])
        rmse_test = []
        rmse_train = []
        nll_val = []
        nll_val_test = []
        for o, el in enumerate(layer_missing_index):
            ind_nan_dim = ind_nan[:, el]
            if ind_nan_dim.sum() > 0:
                mean_test = mean_pred[o][:, ind_nan_dim, :].mean(0).squeeze(-1)
                std_test = std_pred[o][:, ind_nan_dim, :].mean(0)
                y_true_test = y[ind_nan_dim, el]

                rmse_test.append(self.rmse_metric(y_true_test, mean_test).sqrt())
                nll_val_test.append(compute_nn(y_true_test, mean_test, std_test))

                mean_train = mean_pred[o][:, ~ind_nan_dim, :].mean(0).squeeze(-1)
                std_train = std_pred[o][:, ~ind_nan_dim, :].mean(0)
                y_true_train = y[~ind_nan_dim, el]

                rmse_train.append(self.rmse_metric(y_true_train, mean_train).sqrt())
                nll_val.append(compute_nn(y_true_train, mean_train, std_train))

        self.rmse_val = torch.stack(rmse_train).mean()
        self.rmse_val_test = torch.stack(rmse_test).mean()
        self.nll_val_test = torch.stack(nll_val_test).mean()
        self.nll_val = torch.stack(nll_val).mean()

    def logdensity(self, x, mu, var):
        """
        @param x: observed value
        @param mu: prediction
        @param var: prediction variance
        @return: log density normal distribution
        """
        return -0.5 * (np.log(2 * np.pi) + var.log() + (mu - x).square() / var)

    def logp(self, F, Y):
        return self.logdensity(Y, F, self.log_variance.exp())

    def predict_mean_and_var(self, Fmu, Fvar):
        return Fmu, Fvar + self.log_variance.exp()

    def predict_density(self, Fmu, Fvar, Y):
        return self.logdensity(Y, Fmu, Fvar + self.log_variance.exp())

    def variational_expectations(self, Fmu, Fvar, Y):
        """
        @param Fmu: mean prediction
        @param Fvar: variance prediciton
        @param Y: observed values
        @return: variational expection Gaussian likelihood
        """
        return (
            -0.5 * np.log(2 * np.pi)
            - 0.5 * self.log_variance
            - (0.5 / (self.log_variance.exp()))
            * (Y.pow(2) - 2 * Y * Fmu + (Fmu.pow(2) + Fvar))
        ).sum(-1)
        # return -0.5 * np.log(2 * np.pi) - 0.5 * self.log_variance \
        #      - 0.5 * ((Y - Fmu).square() + Fvar) / self.log_variance.exp()
