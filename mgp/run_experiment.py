import torch

import numpy as np

import warnings
from scipy.cluster.vq import kmeans2
import sys

from mgp.model.data import customLoader
from mgp.model.data.preprocess import load_dataset
from mgp.model.kernels import SquaredExponential as RBF
from mgp.model.kernels import White, Matern

from mgp.model.likelihoods import Gaussian

from mgp.model.options import options
from mgp.model.train import train

from mgp.model.train import evaluate
from mgp.model.utils import save_model, load_model
from mgp.model.layers import SVGPLayer
import matplotlib.pyplot as plt
from torch.nn import ModuleList
from sklearn.cluster import KMeans
import random
from mgp.model.dgp import DGP

seed_value = 0
# 1. Set the `PYTHONHASHSEED` environment variable at a fixed value
import os

np.random.seed(0)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


opt = options().parse()
method = opt.name.lower()

if method == "svgp":
    layer_used = SVGPLayer
else:
    layer_used = None


if opt.kernel == "matern":
    kernel_used = Matern
elif opt.kernel == "rbf":
    kernel_used = RBF


if torch.cuda.is_available() and opt.nGPU >= 0:
    device = torch.device("cuda:{}".format(opt.nGPU))
    torch.cuda.set_device(opt.nGPU)
    torch.backends.cudnn.benchmark = True
else:
    device = torch.device("cpu")

print("devcie :{}".format(device.type))

initial_seed = True
if initial_seed:
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
    seed_value = 0
    os.environ["PYTHONHASHSEED"] = str(seed_value)
    # os.environ["CUBLAS_WORKSPACE_CONFIG"] = ':4096:8'
    seed = 0

    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.cuda.manual_seed(seed)
        # torch.backends.cudnn.benchmark = False

    np.random.seed(seed)
    # torch.use_deterministic_algorithms(True)


def compute_nn(y_t, mean_t, std_t):
    f1 = np.log(std_t**2) / 2
    f2 = (y_t - mean_t) ** 2 / (2 * std_t**2)
    f3 = np.log(2 * np.pi) / 2
    # print(std_t.mean())
    return np.mean(f1 + f2 + f3)


def compute_nn_mse(y, mean_pred, std_pred, ind_nan_target):
    D = ind_nan_target.shape[1]
    nll, rmse = [], []
    for d in range(D):
        if ind_nan_target[:, d].sum() > 0:
            a = np.sqrt(
                np.mean(
                    (y[ind_nan_target[:, d], d] - mean_pred[ind_nan_target[:, d], d])
                    ** 2
                )
            )

            b = compute_nn(
                y[ind_nan_target[:, d], d],
                mean_pred[ind_nan_target[:, d], d],
                std_pred[ind_nan_target[:, d], d],
            )

            nll.append(b)
            rmse.append(a)

    nll = np.mean(nll)
    rmse = np.mean(rmse)
    # return nll[~nll.isnan()].sum(), rmse[~rmse.isnan()].sum()
    return rmse, nll


def build_model_missing(
    opt,
    X_train=None,
    y_train=None,
    y_mean=None,
    y_std=None,
    ind_total_nan=None,
    x_means=None,
    x_stds=None,
    ind_total_nan_test=None,
    ind_binary=None,
    scaler=None,
):
    empty_cluster = True
    n_rep = 0
    while empty_cluster:
        try:
            n_rep += 1
            if n_rep > 20:
                ind_sel = np.random.choice(X_train.shape[0], opt.M2)
                Z = X_train[ind_sel, :]
                break
            # Z2 = kmeans2(X_train, opt.M2, minit='points')[0]
            kmeans = KMeans(n_clusters=opt.M2, random_state=42, n_init=10).fit(X_train)
            Z = kmeans.cluster_centers_
            empty_cluster = False
        except Warning:
            pass
    if X_train.shape[0] < 1000:
        value = np.expand_dims(np.sum(np.square(X_train), 1), 1)
        distance = (
            value - 2 * np.matmul(X_train, np.transpose(X_train)) + np.transpose(value)
        )
        median = np.percentile(distance, q=50)
    else:
        median = 10

    lengthscale = np.log(median)

    # We construct the network
    D = X_train.shape[1]

    total_mis = ind_total_nan.sum(0)
    np.where(total_mis > 0)
    args = np.argsort(
        -total_mis[np.where(total_mis > 0)[0]]
    )  # sort from minimum to maximum missing
    layers_missing = np.where(total_mis > 0)[0][
        args
    ]  # np.where(ind_total_nan.sum(0)>0)[0]
    print("before", layers_missing)
    ind_std = np.argsort(x_stds[layers_missing])
    layers_missing = layers_missing[ind_std]
    print("after", layers_missing)
    num_data = []
    global layer_missing_index
    layer_missing_index = []
    num_layers_missing = layers_missing.shape[0]
    if opt.missing:
        num_data = []
        layer_sizes = []
        likelihood_gaussian = ModuleList()
        kernels = torch.nn.ModuleList()
        for layer in range(0, num_layers_missing):
            nD = D - 1
            layer_missing_index.append(layers_missing[layer])
            num_data.append((~ind_total_nan[:, layers_missing[layer]]).sum())
            # num_data.append(X_train.shape[0])

            kernels.append(
                kernel_used(
                    lengthscales=np.tile(lengthscale, nD), variance=1, device=device
                )
                + White(variance=opt.var_noise, device=device)
            )
            layer_sizes.append(nD)
            likelihood_gaussian.append(Gaussian(opt.likelihood_var, device=device))

        X_train = torch.from_numpy(X_train).to(torch.float64).to(device)
        y_train = torch.from_numpy(y_train).to(torch.float64).to(device)
        Z = torch.from_numpy(Z).to(torch.float64).to(device)

        model_missing = DGP(
            X_train,
            y_train,
            Z,
            kernels,
            layer_sizes,
            likelihood_gaussian=likelihood_gaussian,
            likelihood_ber=None,
            num_data=np.array(num_data),
            num_samples=opt.n_samples,
            y_mean=y_mean,
            y_std=y_std,
            warm_up_iters=0,
            device=device,
            ind_total_nan=ind_total_nan,
            layer_missing_index=layer_missing_index,
            ind_binary=ind_binary,
            x_means=x_means,
            x_stds=x_stds,
            scaler=scaler,
            layer=layer_used,
            use_missing_gp=True,
        )
        return model_missing


def build_model(
    opt,
    X_train=None,
    y_train=None,
    y_mean=None,
    y_std=None,
    ind_total_nan=None,
    x_means=None,
    x_stds=None,
    ind_total_nan_test=None,
    ind_binary=None,
    scaler=None,
):
    if opt.minibatch_size is None:
        opt.minibatch_size = min(1e4, X_train.shape[0])
    if opt.nolayers == 1:
        opt.n_samples = 1
    warnings.filterwarnings("error")
    if opt.fitting:
        empty_cluster = True
        n_rep = 0
        while empty_cluster:
            try:
                n_rep += 1
                if n_rep > 20:
                    ind_sel = np.random.choice(X_train.shape[0], opt.M)
                    Z = X_train[ind_sel, :]
                    break

                kmeans = KMeans(n_clusters=opt.M, random_state=0).fit(X_train)
                Z = kmeans.cluster_centers_
                empty_cluster = False
            except Warning:
                pass

    else:
        ind_sel = np.random.choice(X_train.shape[0], opt.M)
        Z = X_train[ind_sel, :]

    warnings.resetwarnings()

    if X_train.shape[0] < 1000:
        value = np.expand_dims(np.sum(np.square(X_train), 1), 1)
        distance = (
            value - 2 * np.matmul(X_train, np.transpose(X_train)) + np.transpose(value)
        )
        median = np.percentile(distance, q=50)
    else:
        median = 10

    lengthscale = np.log(median)

    # We construct the network
    D = X_train.shape[1]

    total_mis = ind_total_nan.sum(0)
    np.where(total_mis > 0)
    args = np.argsort(
        -total_mis[np.where(total_mis > 0)[0]]
    )  # sort form minimum missing to maximum missing
    layers_missing = np.where(total_mis > 0)[0][
        args
    ]  # np.where(ind_total_nan.sum(0)>0)[0]
    ind_std = np.argsort(x_stds[layers_missing])
    layers_missing = np.arange(X_train.shape[1])
    num_data = []
    global layer_missing_index

    layer_missing_index = []
    num_layers_missing = 0

    if method == "nn":
        layer_used = SVGP_NN
    elif method == "svgp":
        layer_used = SVGPLayer
    else:
        layer_used = None

    model_missing = None
    if method != "nn":
        num_data = []
        layer_sizes = []
        likelihood_gaussian = ModuleList()
        kernels = torch.nn.ModuleList()
        for layer in range(0, opt.nolayers):
            nD = D

            layer_missing_index.append(layers_missing[layer])
            # num_data.append((~ind_total_nan[:, layers_missing[layer]]).sum())
            # num_data.append(X_train.shape[0])

            kernels.append(
                kernel_used(
                    lengthscales=np.tile(lengthscale, nD), variance=100, device=device
                )
                + White(variance=opt.var_noise, device=device)
            )
            layer_sizes.append(nD)
        num_data = [X_train.shape[0]]
        # for layer in range(num_layers_missing):
        #   if layer < num_layers_missing:
        #      layer_missing_index.append(layers_missing[layer])
        #     num_data.append((~ind_total_nan[:, layers_missing[layer]]).sum())
        likelihood_gaussian.append(Gaussian(opt.likelihood_var, device=device))
        X_train = torch.from_numpy(X_train).to(torch.float64).to(device)
        y_train = torch.from_numpy(y_train).to(torch.float64).to(device)
        Z = torch.from_numpy(Z).to(torch.float64).to(device)
        # options().parse()
        model = DGP(
            X_train,
            y_train,
            Z,
            kernels,
            layer_sizes,
            likelihood_gaussian=likelihood_gaussian,
            likelihood_ber=None,
            num_data=np.array(num_data),
            num_samples=opt.n_samples,
            y_mean=y_mean,
            y_std=y_std,
            warm_up_iters=0,
            device=device,
            ind_total_nan=ind_total_nan,
            layer_missing_index=layer_missing_index,
            ind_binary=ind_binary,
            x_means=x_means,
            x_stds=x_stds,
            scaler=scaler,
            layer=layer_used,
            use_missing_gp=False,

        )

    else:
        raise NotImplementedError

    return model


def launch_experiment(opt):
    likelihoods, rmses, times = [], [], []

    [
        X_train,
        X_test,
        ind_total_nan,
        imp,
        ind_binary,
        x_means,
        x_stds,
        ind_total_nan_test,
        scaler,
        y_train,
        y_test,
        y_mean,
        y_std,
        X_train_full,
        X_test_full,
    ] = load_dataset(opt)
    if opt.missing:
        add = "missing"
    else:
        add = "non_missing"

    if opt.missing:
        training_generator_missing = customLoader(
            X_train,
            X_train_full,
            ind_total_nan,
            ind_total_nan,
            opt.minibatch_size,
            numThreads=opt.numThreads,
            shuffle=True,
            device=device,
        )

        test_generator_missing = customLoader(
            X_test,
            X_test_full,
            ind_total_nan_test,
            ind_total_nan_test,
            opt.minibatch_size,
            numThreads=opt.numThreads,
            shuffle=False,
            device=device,
        )

    # test_generator = customLoader(X_test, None, ind_total_nan_test, opt.minibatch_size,
    #                             numThreads=opt.numThreads, shuffle=False, device=device)
    path_results = "results/{}/{}/s{}_{}_{}_{}/".format(
        opt.name.lower(),
        opt.dataset_name,
        opt.split_number,
        add,
        opt.imputation,
        opt.nolayers,
    )

    N = X_train.shape[0]
    if not os.path.exists(path_results):
        os.makedirs(path_results)

    file_name = "results.txt"

    global net
    global net_missing
    if opt.missing:
        net_missing = build_model_missing(
            opt,
            X_train=X_train,
            y_train=X_train,
            y_mean=torch.from_numpy(np.array(x_means)).to(device).to(torch.float64),
            y_std=torch.from_numpy(np.array(x_stds)).to(device).to(torch.float64),
            ind_total_nan=ind_total_nan,
            x_means=x_means,
            x_stds=x_stds,
            ind_total_nan_test=ind_total_nan_test,
            ind_binary=ind_binary,
            scaler=scaler,
        )

        params = list(net_missing.parameters())
        print(
            "There are {} trainable parameters for missing value".format(
                np.sum([par.numel() for par in params])
            )
        )
        optimizer = torch.optim.Adam(
            net_missing.parameters(), lr=opt.lrate, weight_decay=0.00
        )
        path_file_name = os.path.join(path_results, file_name)
        n_data_points = X_train.shape[0]
        iter_per_epoch = n_data_points / opt.minibatch_size
        epoch = int(np.ceil(opt.no_iterations / iter_per_epoch))

        X_train = np.tile(np.expand_dims(X_train, 0), [opt.n_samples, 1, 1])
        X_test = np.tile(np.expand_dims(X_test, 0), [opt.n_samples, 1, 1])

        training_generator_missing = customLoader(
            X_train,
            X_train_full,
            ind_total_nan,
            ind_total_nan,
            opt.minibatch_size,
            numThreads=opt.numThreads,
            shuffle=True,
            device=device,
        )

        test_generator_missing = customLoader(
            X_test,
            X_test_full,
            ind_total_nan_test,
            ind_total_nan_test,
            opt.minibatch_size,
            numThreads=opt.numThreads,
            shuffle=False,
            device=device,
        )
        if opt.fitting:
            loggs = train(
                net_missing,
                training_generator_missing,
                test_generator_missing,
                optimizer,
                path_file_name,
                opt.minibatch_size,
                epochs=epoch,
                device=device,
                layer_missing_index=layer_missing_index,
                lrate=opt.lrate,
                ind_binary=ind_binary,
                predict_test=False,
                verbose=True
            )
        else:
            load_model(
                net_missing,
                optimizer,
                path_results,
                which_epoch="latest_missing",
                device=device,
            )
            loggs = train(
                net_missing,
                training_generator_missing,
                test_generator_missing,
                optimizer,
                path_file_name,
                opt.minibatch_size,
                epochs=0,
                device=device,
                layer_missing_index=layer_missing_index,
                lrate=opt.lrate,
                ind_binary=ind_binary,
                predict_test=False,
            )
        training_generator_missing = customLoader(
            X_train,
            X_train_full,
            ind_total_nan,
            ind_total_nan,
            opt.minibatch_size,
            numThreads=opt.numThreads,
            shuffle=True,
            device=device,
        )

        test_generator_missing = customLoader(
            X_test,
            X_test_full,
            ind_total_nan_test,
            ind_total_nan_test,
            opt.minibatch_size,
            numThreads=opt.numThreads,
            shuffle=False,
            device=device,
        )
        (
            NELBO_train,
            NLL_train,
            ACC_train,
            NELBO_test,
            NLL_test,
            ACC_test,
            train_info,
            test_info,
        ) = evaluate(
            net_missing,
            training_generator_missing,
            test_generator_missing,
            path_file_name,
            loggs,
            batch_size=opt.minibatch_size,
            device=device,
        )
        [
            x_aggr_train,
            y_aggr_train,
            mean_pred_aggr_train,
            std_pred_aggr_train,
            ind_nan_aggr_train,
        ] = train_info
        [
            x_aggr_test,
            y_aggr_test,
            mean_pred_aggr_test,
            std_pred_aggr_test,
            ind_nan_aggr_test,
        ] = test_info
        # print(y_mst_test.shape)
        X_test_0 = y_aggr_test
        # y_mst_test_0 = np.mean(y_mst_test, 0)
        rmse_test, nll_test = compute_nn_mse(
            X_test_0,
            np.mean(mean_pred_aggr_test, 0),
            np.mean(std_pred_aggr_test, 0),
            ind_nan_aggr_test,
        )
        print("new_rmse {}, new nll {}".format(rmse_test, nll_test))

        X_train = x_aggr_train * x_stds + x_means
        X_train_full = y_aggr_train * x_stds + x_means
        X_test = x_aggr_test * x_stds + x_means
        X_test_full = y_aggr_test * x_stds + x_means
        # print(y_mst_train[:,0].shape)
        ind_total_nan = ind_nan_aggr_train
        ind_total_nan_test = ind_nan_aggr_test
        s, n, d = mean_pred_aggr_train.shape
        y_mst_train_conv = (
            mean_pred_aggr_train + np.random.randn(s, n, d) * std_pred_aggr_train
        )
        X_train[:, ind_total_nan] = y_mst_train_conv[:, ind_total_nan]

        s, n, d = mean_pred_aggr_test.shape
        y_mst_test_conv = (
            mean_pred_aggr_test + np.random.randn(s, n, d) * std_pred_aggr_test
        )
        X_test[:, ind_total_nan_test] = y_mst_test_conv[:, ind_total_nan_test]

        change_mean_std = False
        if change_mean_std:
            X_t = np.mean(X_train, 0)
            x_means = np.mean(X_t, 0)
            x_stds = X_t.std(0)
            net_missing.x_means = x_means
            net_missing.y_mean = (
                torch.from_numpy(np.array(x_means)).to(device).to(torch.float64)
            )
            net_missing.y_std = (
                torch.from_numpy(np.array(x_stds)).to(device).to(torch.float64)
            )
            net_missing.x_stds = x_stds

        # print(x_stds)
        X_train = (X_train - x_means) / x_stds
        X_test = (X_test - x_means) / x_stds
        X_train_full = (X_train_full - x_means) / x_stds
        X_test_full = (X_test_full - x_means) / x_stds
    else:
        net = build_model(
            opt,
            X_train=X_train,
            y_train=X_train,
            y_mean=torch.from_numpy(np.array(x_means)).to(device).to(torch.float64),
            y_std=torch.from_numpy(np.array(x_stds)).to(device).to(torch.float64),
            ind_total_nan=ind_total_nan,
            x_means=x_means,
            x_stds=x_stds,
            ind_total_nan_test=ind_total_nan_test,
            ind_binary=ind_binary,
            scaler=scaler,
        )
        params = list(net.parameters())
        print(
            "There are {} trainable parameters for missing value".format(
                np.sum([par.numel() for par in params])
            )
        )
        optimizer = torch.optim.Adam(net.parameters(), lr=opt.lrate, weight_decay=0.00)
        path_file_name = os.path.join(path_results, file_name)
        n_data_points = X_train.shape[0]
        iter_per_epoch = n_data_points / opt.minibatch_size
        epoch = int(np.ceil(opt.no_iterations / iter_per_epoch))

        X_train = np.tile(np.expand_dims(X_train, 0), [opt.n_samples, 1, 1])
        X_test = np.tile(np.expand_dims(X_test, 0), [opt.n_samples, 1, 1])
        training_generator_missing = customLoader(
            X_train,
            X_train_full,
            ind_total_nan,
            ind_total_nan,
            opt.minibatch_size,
            numThreads=opt.numThreads,
            shuffle=True,
            device=device,
        )

        test_generator_missing = customLoader(
            X_test,
            X_test_full,
            ind_total_nan_test,
            ind_total_nan_test,
            opt.minibatch_size,
            numThreads=opt.numThreads,
            shuffle=False,
            device=device,
        )
        if opt.fitting:
            loggs = train(
                net,
                training_generator_missing,
                test_generator_missing,
                optimizer,
                path_file_name,
                opt.minibatch_size,
                epochs=epoch,
                device=device,
                layer_missing_index=layer_missing_index,
                lrate=opt.lrate,
                ind_binary=ind_binary,
                predict_test=False,
                verbose=True
            )
        else:
            load_model(
                net, optimizer, path_results, which_epoch="latest", device=device
            )
            loggs = train(
                net,
                training_generator_missing,
                test_generator_missing,
                optimizer,
                path_file_name,
                opt.minibatch_size,
                epochs=0,
                device=device,
                layer_missing_index=layer_missing_index,
                lrate=opt.lrate,
                ind_binary=ind_binary,
                predict_test=False,
            )

        training_generator_missing = customLoader(
            X_train,
            X_train_full,
            ind_total_nan,
            ind_total_nan,
            opt.minibatch_size,
            numThreads=opt.numThreads,
            shuffle=True,
            device=device,
        )

        test_generator_missing = customLoader(
            X_test,
            X_test_full,
            ind_total_nan_test,
            ind_total_nan_test,
            opt.minibatch_size,
            numThreads=opt.numThreads,
            shuffle=False,
            device=device,
        )
        (
            NELBO_train,
            NLL_train,
            ACC_train,
            NELBO_test,
            NLL_test,
            ACC_test,
            train_info,
            test_info,
        ) = evaluate(
            net,
            training_generator_missing,
            test_generator_missing,
            path_file_name,
            loggs,
            batch_size=opt.minibatch_size,
            device=device,
        )
        [
            x_aggr_train,
            y_aggr_train,
            mean_pred_aggr_train,
            std_pred_aggr_train,
            ind_nan_aggr_train,
        ] = train_info
        [
            x_aggr_test,
            y_aggr_test,
            mean_pred_aggr_test,
            std_pred_aggr_test,
            ind_nan_aggr_test,
        ] = test_info
        print(ind_total_nan_test.shape)
        # y_mst_test_0 = np.mean(y_mst_test, 0)
        X_test_0 = y_aggr_test
        rmse_test, nll_test = compute_nn_mse(
            X_test_0,
            np.mean(mean_pred_aggr_test, 0),
            np.mean(std_pred_aggr_test, 0),
            ind_nan_aggr_test,
        )
        print("new_rmse {}, new nll {}".format(rmse_test, nll_test))


launch_experiment(opt)
