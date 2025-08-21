import torch.nn as nn
import numpy as np
import torch
from mgp.model.utils import reparameterize


default_jitter = 1e-7


class Layer(nn.Module):
    def __init__(self, input_prop_dim=None, dtype=None, device=None, **kwargs):
        """ """
        super().__init__()
        self.device = device
        self.dtype = dtype
        self.input_prop_dim = input_prop_dim

    def conditional_ND(self, X, full_cov=False):
        raise NotImplementedError

    def KL(self):
        pass

    def conditional_SND(self, X, full_cov=False, ind_nan=None):
        """
        @param X: input matrix
        @param full_cov: boolean full covariance
        @param ind_nan: indices of missing values
        @return: multisample conditional mean and variance
        """
        if not full_cov:
            S, N, D = X.shape

            X_flat = X.reshape([S * N, D])
            mean, var = self.conditional_ND(X_flat, ind_nan=ind_nan)
            # num_outputs = tf.shape(mean)[-1]
            num_outputs = self.num_outputs
            return [m.reshape([S, N, num_outputs]) for m in [mean, var]]

    def sample_from_conditional(
        self,
        X,
        z=None,
        full_cov=False,
        max_use=None,
        ind_binary=None,
        training=True,
        ind_nan=None,
    ):
        """
        @param X: input matrix
        @param z: matrix for sampling
        @param full_cov: boolean
        @param max_use: boolean
        @param ind_binary:
        @param training: boolean
        @param ind_nan: indice missing values
        @return:
        """
        mean, var = self.conditional_SND(X, full_cov=full_cov, ind_nan=ind_nan)

        # set shapes
        S, N, _ = X.shape
        D = mean.shape[-1]

        mean = mean.reshape((S, N, D))
        if full_cov:
            var = var.reshape((S, N, N, D))
        else:
            var = var.reshape((S, N, D))

        if z is None:
            z = torch.randn((mean.shape), dtype=self.dtype, device=self.device)

        samples = reparameterize(mean, var, z, full_cov=full_cov)

        return samples, mean, var


class SVGPLayer(Layer):
    def __init__(
        self,
        kern,
        Z,
        num_outputs,
        mean_function,
        white=False,
        input_prop_dim=None,
        seed=0,
        dtype=torch.float64,
        device=None,
        ind_total_nan=None,
        n_layers=None,
    ):
        """ """
        super().__init__(input_prop_dim=input_prop_dim, dtype=dtype, device=device)
        self.seed = seed

        self.num_inducing = Z.shape[0]

        # Inducing points prior mean
        self.register_parameter(
            "q_mu",
            nn.Parameter(
                torch.tensor(
                    np.zeros((self.num_inducing, num_outputs)),
                    dtype=self.dtype,
                    device=self.device,
                )
            ),
        )

        self.register_parameter(
            "inducing_points", nn.Parameter(Z.clone().detach().requires_grad_(True))
        )

        self.kern = kern
        self.mean_function = mean_function

        self.num_outputs = num_outputs

        self.white = white
        self.ind_total_nan = ind_total_nan

        if not self.white:  # initialize to prior
            Ku = self.kern.K(Z)
            Lu = torch.linalg.cholesky(
                Ku + torch.eye(Z.shape[0]).to(self.device) * default_jitter
            )
            Lu_tiled = torch.tile(Lu[None, :, :], [num_outputs, 1, 1])
            li, lj = torch.tril_indices(self.num_inducing, self.num_inducing, 0)
            triangular_Lu = Lu_tiled[:, li, lj]

            self.register_parameter(
                "q_sqrt_tri",
                nn.Parameter(triangular_Lu.clone().detach().requires_grad_(True)),
            )

    def get_prior_matrices(self):
        """ " """
        Ku = self.kern.K(self.inducing_points)
        Lu = torch.linalg.cholesky(Ku)  # Ku.cholesky()
        Ku_tiled = torch.tile(Ku[None, :, :], [self.num_outputs, 1, 1])
        Lu_tiled = torch.tile(Lu[None, :, :], [self.num_outputs, 1, 1])

        return Ku, Lu, Ku_tiled, Lu_tiled

    def conditional_ND(self, X, full_cov=False, y=None, ind_nan=None):
        """
        q(f|m, S; X, Z)
        @param X:
        @param full_cov:
        @param y:
        @param ind_nan:
        @return:
        """

        _, Lu, Ku_tiled, _ = self.get_prior_matrices()

        Kuf = self.kern(self.inducing_points, X)

        A = torch.linalg.solve_triangular(Lu, Kuf, upper=False)

        if not self.white:
            A = torch.linalg.solve_triangular(Lu.transpose(0, 1), A, upper=True)

        mean = A.transpose(0, 1).matmul(self.q_mu)

        A_tiled = A[None, :, :].repeat([self.num_outputs, 1, 1])
        I = torch.eye(self.num_inducing)[None, :, :].to(self.dtype).to(self.device)

        if self.white:
            SK = -I
        else:
            SK = -Ku_tiled

        if self.q_sqrt_tri is not None:
            qs = (
                torch.zeros((self.num_inducing, self.num_inducing))
                .to(self.dtype)
                .to(self.device)
            )
            li, lj = torch.tril_indices(self.num_inducing, self.num_inducing, 0)
            qs[li, lj] = self.q_sqrt_tri[0]
            q_sqrt = qs.unsqueeze(0)
            SK += q_sqrt.matmul(q_sqrt.transpose(1, 2))

        B = SK.matmul(A_tiled)

        if not full_cov:
            # (num_latent, num_X)
            delta_cov = (A_tiled * B).sum(1)
            Kff = self.kern.K_diag(X)

        var = Kff.unsqueeze(0) + delta_cov

        return mean + self.mean_function(X, ind_nan), var.transpose(0, 1)

    def KL(self):
        """
        KL DIVERGENCE COMPUTATION
        @return:
        """
        _, Lu, _, Lu_tiled = self.get_prior_matrices()

        D = self.num_outputs

        q_sqrt = (
            torch.zeros((self.num_inducing, self.num_inducing))
            .to(self.dtype)
            .to(self.device)
        )
        li, lj = torch.tril_indices(self.num_inducing, self.num_inducing, 0)
        q_sqrt[li, lj] = self.q_sqrt_tri[0].unsqueeze(0)

        # constant
        KL = -0.5 * D * self.num_inducing

        KL -= 0.5 * q_sqrt.diagonal(0, -2, -1).square().log().sum()

        if not self.white:
            KL += Lu.diagonal(0, -2, -1).log().sum() * D

            KL += (
                0.5
                * torch.linalg.solve_triangular(Lu_tiled, q_sqrt, upper=False)
                .square()
                .sum()
            )

            Kinv_m = torch.cholesky_solve(self.q_mu, Lu)
            KL += 0.5 * (self.q_mu * Kinv_m).sum()
        else:
            KL += 0.5 * q_sqrt.square().sum()
            KL += 0.5 * self.q_mu.square().sum()

        return KL
