import numpy as np

import torch.nn as nn
from mgp.model.mean_functions import Identity, Zero
from mgp.model.layers import SVGPLayer


def init_layers_linear(
    X,
    Y,
    Z,
    kernels,
    layer_sizes,
    mean_function=None,
    num_outputs=None,
    Layer=SVGPLayer,
    whiten=True,
    num_latent=None,
    num_samples=None,
    shared_nn=False,
    device=None,
    ind_total_nan=None,
    layer_missing_index=None,
    opt=None,
):
    """
    initialization of the DGP layers
    @param X:
    @param Y:
    @param Z:
    @param kernels:
    @param layer_sizes:
    @param mean_function:
    @param num_outputs:
    @param Layer:
    @param whiten:
    @param num_latent:
    @param num_samples:
    @param shared_nn:
    @param device:
    @param ind_total_nan:
    @param layer_missing_index:
    @param opt:
    @return:
    """

    kernels_used = kernels

    layers = nn.ModuleList()
    X_running, Z_running = X.clone().detach(), Z.clone().detach().detach()
    if mean_function is None:
        mean_function = Zero(device=device).to(device)

    in_idx = 0
    for kern_in in kernels_used:
        dim_in = layer_sizes[in_idx]
        dim_out = dim_in

        Z_used = Z_running
        X_used = X_running

        # Initialize mean function to be either Identity or PCA projection
        if dim_in == dim_out and in_idx != len(kernels) - 1:
            mf = Identity(device=device).to(device)
        else:
            mf = Zero(device=device).to(device)
        layer = Layer(kern_in, Z_used, dim_out, mf, white=whiten, device=device)
        layers.append(layer)
        in_idx += 1

    return layers


def init_layers_linear_missing(
    X,
    Y,
    Z,
    kernels,
    layer_sizes,
    mean_function=None,
    num_outputs=None,
    Layer=SVGPLayer,
    whiten=True,
    num_latent=None,
    num_samples=None,
    shared_nn=False,
    device=None,
    ind_total_nan=None,
    layer_missing_index=None,
    opt=None,
):
    """
    initialization of the DGP layers for MGP
    @param X:
    @param Y:
    @param Z:
    @param kernels:
    @param layer_sizes:
    @param mean_function:
    @param num_outputs:
    @param Layer:
    @param whiten:
    @param num_latent:
    @param num_samples:
    @param shared_nn:
    @param device:
    @param ind_total_nan:
    @param layer_missing_index:
    @param opt:
    @return:
    """
    kernels_used = kernels
    total_dim = X.shape[1]
    layers = nn.ModuleList()
    X_running, Z_running = X.clone().detach(), Z.clone().detach().detach()
    in_idx = 0
    for kern_in in kernels_used:
        layer_miss_ind = layer_missing_index[in_idx]
        dim_out = 1  # layer_sizes[in_idx + 1]
        dim_in = layer_sizes[in_idx]

        Z_used = Z_running
        mf = Zero(device=device).to(device)
        other_inds = np.array((list(set(np.arange(total_dim)) - set({layer_miss_ind}))))
        layer = Layer(
            kern_in, Z_used[:, other_inds], dim_out, mf, white=whiten, device=device
        )

        layers.append(layer)
        in_idx += 1

    return layers
