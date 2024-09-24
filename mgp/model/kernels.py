# Credit to GPflow

import torch
import torch.nn as nn
import numpy as np
from torch.nn.functional import softmax
import abc
from functools import partial, reduce
from typing import List, Optional, Union

ActiveDims = Union[slice, list]


def square_distance(X, X2):
    """
    RBF KERNEL
    """

    Xs = X.square().sum(-1).unsqueeze(-1)
    if X2 is None:
        dist = -2 * X.matmul(X.transpose(-1, -2))

        dist += Xs + Xs.transpose(-1, -2)
        return dist

    X2s = X2.square().sum(-1).unsqueeze(-1)
    dist = -2 * X.matmul(X2.transpose(-1, -2))
    dist += Xs + X2s.transpose(-1, -2)
    return dist


class Kernel(nn.Module):
    """
    base class for kernels
    """

    def name(self):
        "Kernel"

    def __init__(
        self,
        active_dims: Optional[ActiveDims] = None,
        name: Optional[str] = None,
        dtype=torch.float64,
        device=None,
    ):
        """ """
        super().__init__()
        self.device = device
        self.dtype = dtype
        self._active_dims = self._normalize_active_dims(active_dims)

    @staticmethod
    def _normalize_active_dims(value):
        if value is None:
            value = slice(None, None, None)
        if not isinstance(value, slice):
            value = np.array(value, dtype=int)
        return value

    @property
    def active_dims(self):
        return self._active_dims

    @active_dims.setter
    def active_dims(self, value):
        self._active_dims = self._normalize_active_dims(value)

    def slice(self, X: torch.Tensor, X2: Optional[torch.Tensor] = None):
        """ """
        dims = self.active_dims
        if isinstance(dims, slice):
            X = X[..., dims]
            if X2 is not None:
                X2 = X2[..., dims]

        return X, X2

    def _validate_ard_active_dims(self, ard_parameter):
        """
        Validate that ARD parameter matches the number of active_dims (provided active_dims
        has been specified as an array).
        """
        if self.active_dims is None or isinstance(self.active_dims, slice):
            # Can only validate parameter if active_dims is an array
            return

        if ard_parameter.shape.rank > 0 and ard_parameter.shape[0] != len(
            self.active_dims
        ):
            raise ValueError(
                f"Size of `active_dims` {self.active_dims} does not match "
                f"size of ard parameter ({ard_parameter.shape[0]})"
            )

    @abc.abstractmethod
    def K(self, X, X2=None):
        raise NotImplementedError

    @abc.abstractmethod
    def K_diag(self, X):
        raise NotImplementedError

    def __call__(self, X, X2=None, *, full_cov=True, presliced=False):
        if (not full_cov) and (X2 is not None):
            raise ValueError(
                "Ambiguous inputs: `not full_cov` and `X2` are not compatible."
            )

        if not presliced:
            X, X2 = self.slice(X, X2)

        if not full_cov:
            assert X2 is None
            return self.K_diag(X)

        else:
            return self.K(X, X2)

    def __add__(self, other):
        return Sum([self, other])


class Combination(Kernel):
    """ """

    _reduction = None

    def __init__(self, kernels: List[Kernel], name: Optional[str] = None):
        super().__init__(name=name)

        if not all(isinstance(k, Kernel) for k in kernels):
            raise TypeError("can only combine Kernel instances")  # pragma: no cover

        self._set_kernels(kernels)

    def _set_kernels(self, kernels: List[Kernel]):
        kernels_list = nn.ModuleList()
        for k in kernels:
            if isinstance(k, self.__class__):
                kernels_list.extend(k.kernels)
            else:
                kernels_list.append(k)
        self.kernels = kernels_list

    @property
    def on_separate_dimensions(self):
        """ """
        if np.any([isinstance(k.active_dims, slice) for k in self.kernels]):
            return False
        else:
            dimlist = [k.active_dims for k in self.kernels]
            overlapping = False
            for i, dims_i in enumerate(dimlist):
                for dims_j in dimlist[i + 1 :]:
                    print(f"dims_i = {type(dims_i)}")
                    if np.any(dims_i.reshape(-1, 1) == dims_j.reshape(1, -1)):
                        overlapping = True
            return not overlapping


class ReducingCombination(Combination):
    def __call__(self, X, X2=None, *, full_cov=True, presliced=False):
        return self._reduce(
            [k(X, X2, full_cov=full_cov, presliced=presliced) for k in self.kernels]
        )

    def K(self, X: torch.Tensor, X2: Optional[torch.Tensor] = None) -> torch.Tensor:
        return self._reduce([k.K(X, X2) for k in self.kernels])

    def K_diag(self, X: torch.Tensor) -> torch.Tensor:
        return self._reduce([k.K_diag(X) for k in self.kernels])

    @property
    @abc.abstractmethod
    def _reduce(self):
        pass


class Sum(ReducingCombination):
    @property
    def _reduce(self):
        return sum


class Stationary(Kernel):
    """ """

    def __init__(self, variance=1.0, lengthscales=1.0, device=None, **kwargs):
        """ """
        for kwarg in kwargs:
            if kwarg not in {"name", "active_dims"}:
                raise TypeError(f"Unknown keyword argument: {kwarg}")

        super().__init__(device=device, **kwargs)

        self.register_parameter(
            "log_variance",
            nn.Parameter(
                torch.tensor(np.log(variance), dtype=self.dtype, device=self.device)
            ),
        )

        self.register_parameter(
            "log_lengthscales",
            nn.Parameter(
                torch.tensor(np.log(lengthscales), dtype=self.dtype, device=self.device)
            ),
        )

        self._validate_ard_active_dims(self.log_lengthscales)

    @property
    def ard(self) -> bool:
        """ """
        return self.log_lengthscales.shape.ndims > 0

    def scale(self, X):
        X_scaled = X / torch.exp(self.log_lengthscales) if X is not None else X
        return X_scaled

    def K_diag(self, X):
        return (self.log_variance).exp().repeat(X.shape[-2])


class IsotropicStationary(Stationary):
    """ """

    def K(self, X, X2=None):
        r2 = self.scaled_squared_euclid_dist(X, X2)
        return self.K_r2(r2)

    def scaled_squared_euclid_dist(self, X, X2=None):
        """ """
        return square_distance(self.scale(X), self.scale(X2))


class Static(Kernel):
    """ """

    def __init__(self, variance=1.0, active_dims=None, device=None):
        super().__init__(active_dims, device=device)

        self.register_parameter(
            "log_variance",
            nn.Parameter(
                torch.tensor(np.log(variance), dtype=self.dtype, device=self.device)
            ),
        )

    def K_diag(self, X):
        return (self.log_variance).exp().repeat(X.shape[-2])


class White(Static):
    """ """

    def K(self, X, X2=None):
        if X2 is None:
            d = torch.tile((self.log_variance).exp().squeeze(), dims=(X.shape[0], 1))[
                :, 0
            ]
            return torch.diag_embed(d)
        else:
            return torch.zeros((X.shape[-2], X2.shape[-2])).to(X.device).to(X.dtype)


class SquaredExponential(IsotropicStationary):
    """ """

    def K_r2(self, r2):
        return (self.log_variance).exp() * torch.exp(-0.5 * r2)


class Matern(IsotropicStationary):
    """
    @
    """

    def K(self, X, X2=None):
        r = self.scaled_euclid_dist(X, X2)
        return self.K_r(r)

    def scaled_euclid_dist(self, X, X2=None):
        """ """
        if X2 is None:
            return torch.cdist(self.scale(X), self.scale(X), p=2).sqrt()
        return torch.cdist(self.scale(X), self.scale(X2), p=2).sqrt()

    def K_r(self, r, nu=1.5):
        # multiply by term
        exp_c = torch.exp(-np.sqrt(nu * 2) * r)

        if nu == 0.5:
            # Matern 1/2
            const_c = 1
        elif nu == 1.5:
            # Matern 3/2
            const_c = 1.0 + np.sqrt(3) * r
        elif nu == 2.5:
            # Matern 5/2
            const_c = np.sqrt(5) * r + 1.0 + 5.0 / 3.0 * r**2

        return (self.log_variance).exp() * const_c * exp_c
