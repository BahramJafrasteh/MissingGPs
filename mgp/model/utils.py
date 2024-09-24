import torch
import numpy as np

default_jitter = 1e-7
default_float = torch.float64
import os

rmse_metric = torch.nn.MSELoss()


def compute_nn(y_t, mean_t, std_t):
    f1 = torch.log(std_t**2) / 2
    f2 = (y_t - mean_t) ** 2 / (2 * std_t**2)
    f3 = np.log(2 * np.pi) / 2
    # print(std_t.mean())
    return torch.mean(f1 + f2 + f3)


def compute_nn_mse(y, mean_pred, std_pred, ind_nan_target):
    D = ind_nan_target.shape[1]
    nll, rmse = [], []
    for d in range(D):
        if ind_nan_target[:, d].sum() > 0:
            a = rmse_metric(
                y[ind_nan_target[:, d], d],
                mean_pred[:, ind_nan_target[:, d], d].mean(0),
            ).sqrt()
            b = compute_nn(
                y[ind_nan_target[:, d], d],
                mean_pred[:, ind_nan_target[:, d], d].squeeze(-1),
                std_pred[:, ind_nan_target[:, d], d].squeeze(-1),
            )

            nll.append(b)
            rmse.append(a)

    nll = torch.stack(nll)
    rmse = torch.stack(rmse)
    # return nll[~nll.isnan()].sum(), rmse[~rmse.isnan()].sum()
    return nll.mean(), rmse.mean()


def compute_metrics(y, mean_pred, std_pred, ind_nan):
    def compute_nn0(y_t, mean_t, std_t):
        S = mean_t.shape[0]
        normal = torch.distributions.Normal(loc=mean_t, scale=std_t)
        logpdf = normal.log_prob(y_t)
        nll = torch.logsumexp(logpdf, 0) - np.log(S)
        return -nll.mean()

    def compute_nn(y_t, mean_t, std_t):
        # std_t = std_t/std_t.max()
        f1 = torch.log(std_t**2) / 2
        f2 = (y_t - mean_t) ** 2 / (2 * std_t**2)
        f3 = np.log(2 * np.pi) / 2

        return torch.mean(f1 + f2 + f3)

    if ind_nan is not None:
        mean_test = mean_pred[ind_nan]
        std_test = std_pred[ind_nan]
        y_test = y[ind_nan]

        mean_train = mean_pred[~ind_nan]
        std_train = std_pred[~ind_nan]
        y_train = y[~ind_nan]

        nll_test = compute_nn(y_test, mean_test, std_test)
        rmse_test = rmse_metric(y_test, mean_test).sqrt()

        nll_train = compute_nn(y_train, mean_train, std_train)
        rmse_train = rmse_metric(y_train, mean_train).sqrt()
        return rmse_train, nll_train, rmse_test, nll_test
    else:
        nll_test = compute_nn(y, mean_pred, std_pred)
        rmse_test = rmse_metric(y, mean_pred).sqrt()
        return rmse_test, nll_test


def accuracy(out, labels):
    if out.min() < 0:
        outputs = torch.where(out < 0.5, -1.0, 1.0)
    else:
        outputs = torch.where(out < 0.5, 0.0, 1.0)
    return torch.sum(outputs == labels) / (labels.shape[0])


# save models to the disk
def save_model(model, optimizer, device, model_path, epoch="latest"):
    save_filename = "VSGP_%s.pth" % (epoch)
    save_path = os.path.join(model_path, "checkpoints/")
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    save_filepath = os.path.join(model_path, "checkpoints/", save_filename)
    np_state = np.random.get_state()
    if device.type == "cuda" and torch.cuda.is_available():
        trch_rng_state_cuda = torch.cuda.get_rng_state(device).cpu()
        trch_rng_state = torch.get_rng_state().cpu()
        state = {
            "state_dict": model.cpu().state_dict(),
            "optimizer": optimizer.state_dict(),
            "np_state": np_state,
            "trch_state": trch_rng_state,
            "trch_cuda_state": trch_rng_state_cuda,
        }
        torch.save(state, save_filepath)
        model.cuda(device)
    else:
        # trch_rng_state = torch.get_rng_state(device)
        state = {
            "state_dict": model.cpu().state_dict(),
            "optimizer": optimizer.state_dict(),
            "np_state": np_state,
        }
        torch.save(state, save_filepath)


def load_model(model, optimizer, model_path, device, which_epoch="latest"):
    # load models from the disk
    load_filename = "VSGP_%s.pth" % (which_epoch)

    load_path = os.path.join(model_path, "checkpoints/", load_filename)

    print("loading the model from %s" % load_path)
    # if you are using PyTorch newer than 0.4 (e.g., built from
    # GitHub source), you can remove str() on self.device
    state_dict = torch.load(load_path, map_location=device)

    if "state_dict" in state_dict:
        model.load_state_dict(state_dict["state_dict"])
        # optimizer.load_state_dict(state_dict['optimizer'])
        if torch.cuda.is_available() and device.type == "cuda":
            torch.cuda.set_rng_state(state_dict["trch_cuda_state"].cpu())
            torch.set_rng_state(state_dict["trch_state"].cpu())
        np.random.set_state(state_dict["np_state"])
        for state in optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(device)
                else:
                    state[k] = v

    else:
        model.load_state_dict(state_dict)

    model.eval()


def reparameterize(mean, var, z, full_cov=False):
    """
    The 'reparameterization trick' for the Gaussian
    """

    if var is None:
        return mean

    if full_cov is False:
        return mean + z * (var + default_jitter) ** 0.5
