import torch
import torch.nn as nn

from mgp.model.layer_initializations import (
    init_layers_linear,
    init_layers_linear_missing,
)
import numpy as np


class DGP_Base(nn.Module):
    def name(self):
        return "DGP_Base"

    def __init__(
        self,
        likelihood_gaussian,
        likelihood_ber,
        layers,
        num_data,
        num_samples=1,
        y_mean=0.0,
        y_std=1.0,
        warm_up_iters=None,
        dtype=torch.float64,
        seed=0,
        device=None,
        ind_total_nan=None,
        layer_missing_index=None,
        ind_binary=None,
        x_means=None,
        x_stds=None,
        **kwargs
    ):
        super().__init__()
        self.x_means = x_means
        self.x_stds = x_stds
        self.device = device
        self.dtype = dtype
        self.num_samples = num_samples
        self.num_data = num_data
        self.y_mean = y_mean
        self.y_std = y_std
        self.back_transform = False
        self.warm_up_iters = warm_up_iters

        self.likelihood_gaussian = likelihood_gaussian
        self.likelihood_bern = likelihood_ber
        self.layers_dgp = layers
        torch.manual_seed(seed)
        self.use_interconnection = False
        self.max_use = 1
        self.current_dim = 0
        self.ind_total_nan = ind_total_nan
        self.layer_missing_index = layer_missing_index
        self.ind_binary = ind_binary
        self.rmse_metric = torch.nn.MSELoss()
        self.no_gt = False

    def train_step(
        self,
        optimizer,
        batch_x,
        batch_y,
        ind_nan,
        ind_nan_target,
        current_iter,
        max_use,
    ):
        """
        Training step
        @param optimizer:
        @param batch_x:
        @param batch_y:
        @param ind_nan:
        @param ind_nan_target:
        @param current_iter:
        @param max_use:
        @return:
        """
        if batch_y.ndim == 1:
            batch_y = batch_y.unsqueeze(-1)

        if self.dtype != batch_x.dtype:
            batch_x = batch_x.to(self.dtype)
        if self.dtype != batch_y.dtype:
            batch_y = batch_y.to(self.dtype)

        loss = self.nelbo(
            batch_x, batch_y, ind_nan, ind_nan_target, current_iter, max_use=max_use
        )
        with torch.no_grad():
            means, std_pred = self.forward(batch_x, ind_nan=ind_nan, max_use=max_use)
        optimizer.zero_grad()
        loss.backward()  # retain_graph=True

        optimizer.step()
        if self.back_transform:
            self.update_metrics(
                batch_y * self.y_std + self.y_mean,
                means,
                std_pred,
                ind_nan=ind_nan,
                ind_nan_target=ind_nan_target,
                layer_missing_index=self.layer_missing_index,
            )
        else:
            self.update_metrics(
                batch_y,
                means,
                std_pred,
                ind_nan=ind_nan,
                ind_nan_target=ind_nan_target,
                layer_missing_index=self.layer_missing_index,
            )

        return loss

    def update_metrics(
        self,
        y,
        mean_pred,
        std_pred,
        ind_nan=None,
        ind_nan_target=None,
        layer_missing_index=None,
        max_use=None,
    ):
        """
        updating metrics
        @param y:
        @param mean_pred:
        @param std_pred:
        @param ind_nan:
        @param ind_nan_target:
        @param layer_missing_index:
        @param max_use:
        @return:
        """

        def compute_nn(y_t, mean_t, std_t):
            f1 = torch.log(std_t**2) / 2
            f2 = (y_t - mean_t) ** 2 / (2 * std_t**2)
            f3 = np.log(2 * np.pi) / 2

            return torch.mean(f1 + f2 + f3)

        def compute_nn_mse(y, mean_pred, std_pred, ind_nan_target):
            ind_true_target = ~ind_nan_target
            D = ind_nan_target.shape[1]
            nll, rmse = [], []
            for d in range(D):
                if ind_nan_target[:, d].sum() > 0:
                    a = self.rmse_metric(
                        y[ind_true_target[:, d], d],
                        mean_pred[:, ind_true_target[:, d], d].mean(0),
                    ).sqrt()
                    b = compute_nn(
                        y[ind_true_target[:, d], d],
                        mean_pred[:, ind_true_target[:, d], d].squeeze(-1),
                        std_pred[:, ind_true_target[:, d], d].squeeze(-1),
                    )

                    nll.append(b)
                    rmse.append(a)

            nll = torch.stack(nll)
            rmse = torch.stack(rmse)

            return nll.mean(), rmse.mean()

        if not self.onelayer_out:
            self.mean_pred_aggr = torch.stack(mean_pred, dim=2).squeeze(-1).mean(0)
            self.std_pred_aggr = torch.stack(std_pred, dim=2).squeeze(-1).mean(0)
            ind_nan_aggr = []
            y_aggr = []
            for o, el in enumerate(layer_missing_index):
                ind_nan_aggr.append(ind_nan[:, el])
                y_aggr.append(y[:, el])
            self.ind_nan_aggr = torch.stack(ind_nan_aggr, dim=1)
            self.y_aggr = torch.stack(y_aggr, dim=1)

            self.rmse_val = self.rmse_metric(
                self.y_aggr[~self.ind_nan_aggr], self.mean_pred_aggr[~self.ind_nan_aggr]
            ).sqrt()

            self.nll_val = compute_nn(
                self.y_aggr[~self.ind_nan_aggr],
                self.mean_pred_aggr[~self.ind_nan_aggr],
                self.std_pred_aggr[~self.ind_nan_aggr],
            )

            self.rmse_val_test = self.rmse_metric(
                self.y_aggr[self.ind_nan_aggr], self.mean_pred_aggr[self.ind_nan_aggr]
            ).sqrt()
            self.nll_val_test = compute_nn(
                self.y_aggr[self.ind_nan_aggr],
                self.mean_pred_aggr[self.ind_nan_aggr],
                self.std_pred_aggr[self.ind_nan_aggr],
            )

        else:
            self.mean_pred_aggr = mean_pred
            self.std_pred_aggr = std_pred

            if self.use_missing_gp:
                self.y_aggr = y
                ind_nan_target = ind_nan

                self.nll_val, self.rmse_val = compute_nn_mse(
                    y, mean_pred, std_pred, ind_nan_target
                )

                y_t = y * self.y_std + self.y_mean
                mean_pred_t = mean_pred * self.y_std + self.y_mean
                std_pred_t = std_pred * self.y_std
                self.nll_val_converted, self.rmse_val_converted = compute_nn_mse(
                    y_t, mean_pred_t, std_pred_t, ind_nan_target
                )

            else:
                self.y_aggr = y

                self.nll_val, self.rmse_val = compute_nn_mse(
                    y, mean_pred, std_pred, ind_nan_target
                )

                y_t = y * self.y_std + self.y_mean
                mean_pred_t = mean_pred * self.y_std + self.y_mean
                std_pred_t = std_pred * self.y_std
                self.nll_val_converted, self.rmse_val_converted = compute_nn_mse(
                    y_t, mean_pred_t, std_pred_t, ind_nan_target
                )

    def forward(
        self,
        inputs,
        imputing_missing_values=False,
        ind_nan=None,
        max_use=np.inf,
        num_samples=1000,
        training=False,
        back_transform=True,
    ):
        """
        @param inputs: input data
        """

        output_means, output_vars, Ds_predict = self.predict_y(
            inputs,
            self.num_samples,
            ind_nan=ind_nan,
            max_use=max_use,
            training=training,
        )
        if self.use_missing_gp:
            output_means_converted = inputs.clone()
            output_sqrt_converted = torch.zeros_like(output_means_converted)

            for l_gauss, el in enumerate(self.layer_missing_index):
                Fmean = output_means[l_gauss]
                Fvar = output_vars[l_gauss]
                D_predict = Ds_predict[l_gauss]

                Fvar += self.likelihood_gaussian[l_gauss].log_variance.exp()
                if self.back_transform:
                    Fsqrt_converted = Fvar.sqrt() * self.y_std[D_predict]
                    Fmean_converted = (
                        Fmean * self.y_std[D_predict] + self.y_mean[D_predict]
                    )
                else:
                    Fsqrt_converted = Fvar.sqrt()
                    Fmean_converted = Fmean

                output_means_converted[:, :, D_predict] = Fmean_converted.squeeze(-1)
                output_sqrt_converted[:, :, D_predict] = Fsqrt_converted.squeeze(-1)

        else:
            output_vars[-1] += self.likelihood_gaussian[-1].log_variance.exp()
            if self.back_transform:
                output_sqrt_converted = output_vars[-1].sqrt() * self.y_std
                output_means_converted = output_means[-1] * self.y_std + self.y_mean
            else:
                output_sqrt_converted = output_vars[-1].sqrt()
                output_means_converted = output_means[-1]

        return output_means_converted, output_sqrt_converted

    #
    def test_step(self, batch_x, batch_y, ind_nan, ind_nan_target, normalize_y=True):
        """
        testing the method
        @param batch_x:
        @param batch_y:
        @param ind_nan:
        @param ind_nan_target:
        @param normalize_y:
        @return:
        """
        if batch_y.ndim == 1:
            batch_y = batch_y.unsqueeze(-1)

        if self.dtype != batch_x.dtype:
            batch_x = batch_x.to(self.dtype)
        if self.dtype != batch_y.dtype:
            batch_y = batch_y.to(self.dtype)

        # Compute predictions
        with torch.no_grad():
            mean_pred, var_pred = self(
                batch_x, ind_nan=ind_nan, training=False
            )  # Forward pass
            if self.back_transform:
                self.update_metrics(
                    batch_y * self.y_std + self.y_mean,
                    mean_pred,
                    var_pred,
                    ind_nan=ind_nan,
                    ind_nan_target=ind_nan_target,
                    layer_missing_index=self.layer_missing_index,
                )
            else:
                self.update_metrics(
                    batch_y,
                    mean_pred,
                    var_pred,
                    ind_nan=ind_nan,
                    ind_nan_target=ind_nan_target,
                    layer_missing_index=self.layer_missing_index,
                )
            # Compute the loss
            loss = self.nelbo(
                batch_x, batch_y, ind_nan, ind_nan_target, 0, max_use=np.inf
            )

        return loss

    def propagate(
        self,
        X,
        full_cov=False,
        num_samples=1,
        zs=None,
        ind_nan=None,
        max_use=None,
        training=True,
    ):
        """
        computing the results
        @param X:
        @param full_cov:
        @param num_samples:
        @param zs:
        @param ind_nan:
        @param max_use:
        @param training:
        @return:
        """
        if self.use_missing_gp:
            F = X
        else:
            if X.ndim != 3:
                sX = torch.tile(X.unsqueeze(0), [num_samples, 1, 1])
                F = sX
            else:
                F = X.clone()

        zs = zs or [
            None,
        ] * len(self.layers_dgp)

        if max_use != np.inf:
            max_use = len(self.layer_missing_index) - 1

        size_loop = 1

        for iter in range(size_loop):
            Fs, Fmeans, Fvars = [], [], []
            l_n = 0
            Fmean, Fvar = [], []
            Ds_predict = []
            for layer, z in zip(self.layers_dgp, zs):
                if self.use_missing_gp:
                    D_predict = self.layer_missing_index[l_n]
                    effective_dims = list(set(np.arange(F.shape[2])) - set([D_predict]))

                    GG, Fmean, Fvar = layer.sample_from_conditional(
                        F[:, :, effective_dims],
                        z=z,
                        full_cov=full_cov,
                        max_use=max_use,
                        ind_binary=self.ind_binary,
                    )
                    Ds_predict.append(D_predict)
                    F[:, ind_nan[:, D_predict], D_predict] = (
                        GG[:, ind_nan[:, D_predict], :].squeeze(-1).clone()
                    )  # + F[:, :, D_predict].squeeze(-1)#

                else:
                    F, Fmean, Fvar = layer.sample_from_conditional(
                        F, z=z, full_cov=full_cov
                    )

                if iter == size_loop - 1:
                    Fs.append(F)
                    Fmeans.append(Fmean)
                    Fvars.append(Fvar)

                l_n += 1

        return Fs, Fmeans, Fvars, Ds_predict

    def predict_f(
        self,
        predict_at,
        num_samples,
        full_cov=False,
        ind_nan=None,
        max_use=None,
        training=True,
    ):
        Fs, Fmeans, Fvars, Ds_predict = self.propagate(
            predict_at,
            full_cov=full_cov,
            num_samples=num_samples,
            ind_nan=ind_nan,
            max_use=max_use,
            training=training,
        )
        return Fmeans, Fvars, Ds_predict

    def predict_all_layers(self, predict_at, num_samples, full_cov=False):
        return self.propagate(predict_at, full_cov=full_cov, num_samples=num_samples)

    def predict_y(
        self, predict_at, num_samples, ind_nan=None, max_use=None, training=False
    ):
        Fmean, Fvar, Ds_predict = self.predict_f(
            predict_at,
            num_samples=num_samples,
            full_cov=False,
            ind_nan=ind_nan,
            max_use=max_use,
            training=training,
        )

        return (
            Fmean,
            Fvar,
            Ds_predict,
        )  # self.likelihood_gaussian.predict_mean_and_var(Fmean[-1], Fvar[-1])

    def expected_data_log_likelihood(
        self,
        X,
        Y,
        ind_nan,
        ind_nan_target,
        index_used=None,
        max_use=np.inf,
        current_iter=0,
    ):
        """
        Compute expectations of the loglikelihood
        """
        F_means, F_vars, Ds_predict = self.predict_f(
            X,
            num_samples=self.num_samples,
            full_cov=False,
            ind_nan=ind_nan,
            max_use=max_use,
            training=True,
        )
        var_exp = 0

        if self.use_missing_gp:
            X_true = Y
            ind_total_true = ~ind_nan_target

            for l_gauss, el in enumerate(self.layer_missing_index):
                Fmeans = F_means[l_gauss]
                Fvars = F_vars[l_gauss]
                D_predict = Ds_predict[l_gauss]
                scale = (
                    self.num_data[l_gauss] / ind_total_true[:, D_predict].sum()
                )  # ind_total_true[:,o].sum()

                var_exp += (
                    scale
                    * self.likelihood_gaussian[l_gauss]
                    .variational_expectations(
                        Fmeans[:, ind_total_true[:, D_predict]].squeeze(-1),
                        Fvars[:, ind_total_true[:, D_predict]].squeeze(-1),
                        X_true[ind_total_true[:, D_predict], D_predict],
                    )
                    .mean(0)
                    .sum()
                )  # Shape [S, N, D]

        else:
            scale = self.num_data[-1] / X.shape[0]  # ind_total_true[:,o].sum()

            if self.no_gt:
                ind_total_true = ~ind_nan_target
                scale = self.num_data[-1] / ind_total_true.sum()
                var_exp += (
                    scale
                    * self.likelihood_gaussian[-1]
                    .variational_expectations(
                        F_means[-1][:, ind_total_true],
                        F_vars[-1][:, ind_total_true],
                        Y[ind_total_true],
                    )
                    .mean(0)
                    .sum()
                )
            else:
                var_exp += (
                    scale
                    * self.likelihood_gaussian[-1]
                    .variational_expectations(F_means[-1], F_vars[-1], Y)
                    .mean(0)
                    .sum()
                )
        return var_exp  # torch.mean(var_exp, dim=0)  # Shape [N, D]

    #
    def nelbo(
        self, inputs, outputs, ind_nan, ind_nan_target, current_iter, max_use=np.inf
    ):
        """
        Computes the negative elbo
        """
        X, Y = inputs, outputs

        likelihood = self.expected_data_log_likelihood(
            X, Y, ind_nan, ind_nan_target, max_use=max_use, current_iter=current_iter
        )

        if self.use_missing_gp:
            KL = torch.stack(
                [layer.KL() for num, layer in enumerate(self.layers_dgp)]
            ).sum()

        else:
            KL = torch.stack(
                [layer.KL() for num, layer in enumerate(self.layers_dgp)]
            ).sum()

        return -(likelihood - KL)  # + weight_regularizer

        # return - (-  KL)

    @property
    def metrics(self):
        metrics = [self.loss_tracker]
        for metric in self.likelihood_gaussian.metrics:
            metrics.append(metric)

        return metrics


class DGP(DGP_Base):
    """
    Doubly-Stochastic Deep GP, with linear/identity mean functions at each layer.

      @inproceedings{salimbeni2017doubly,
        title={Doubly Stochastic Variational Inference for Deep Gaussian Processes},
        author={Salimbeni, Hugh and Deisenroth, Marc},
        booktitle={NIPS},
        year={2017}
      }

    """

    def __init__(
        self,
        X,
        Y,
        Z,
        kernels,
        layer_sizes,
        likelihood_gaussian,
        likelihood_ber,
        num_data,
        y_mean=0.0,
        y_std=1.0,
        warm_up_iters=None,
        num_outputs=None,
        mean_function=None,
        whiten=False,
        num_samples=1,
        device=None,
        ind_total_nan=None,
        layer_missing_index=None,
        ind_binary=None,
        x_means=None,
        x_stds=None,
        scaler=None,
        layer=None,
        use_missing_gp=True,
    ):
        if use_missing_gp:
            inits = init_layers_linear_missing
        else:
            inits = init_layers_linear
        layers = inits(
            X,
            Y,
            Z,
            kernels,
            layer_sizes,
            mean_function=mean_function,
            num_outputs=num_outputs,
            whiten=whiten,
            device=device,
            ind_total_nan=ind_total_nan,
            layer_missing_index=layer_missing_index,
            Layer=layer,
        )
        self.use_missing_gp = use_missing_gp
        self.scaler = scaler
        self.onelayer_out = False
        if len(layer_sizes) - len(layer_missing_index) > 0:
            self.onelayer_out = True
        self.onelayer_out = True
        super().__init__(
            likelihood_gaussian,
            likelihood_ber,
            layers,
            num_data,
            num_samples,
            y_mean=y_mean,
            y_std=y_std,
            warm_up_iters=warm_up_iters,
            device=device,
            ind_total_nan=ind_total_nan,
            layer_missing_index=layer_missing_index,
            ind_binary=ind_binary,
            x_means=x_means,
            x_stds=x_stds,
        )
