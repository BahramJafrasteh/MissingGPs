from tabnanny import verbose

import torch
import numpy as np
import time

from mgp.model.utils import *
from mgp.model.callback import logger
import pkbar
from torch.optim.lr_scheduler import StepLR


def train(
    model,
    training_generator,
    test_generator,
    optimizer,
    path_file_name,
    batch_size=100,
    epochs=100,
    device=None,
    layer_missing_index=None,
    lrate=0.1,
    ind_binary=None,
    predict_test=True,
    verbose=False,
):
    """
    @param model: model
    @param training_generator: training data
    @param test_generator: test data
    @param optimizer: optimizer
    @param path_file_name: path for writing the output
    @param batch_size: batch size
    @param epochs: number of epochs
    @param device: the used device CPU/GPU
    @param layer_missing_index: missign index for the layers
    @param lrate: learning rate
    @param ind_binary:
    @param predict_test: prediction on test
    @return: trained model
    """

    model.train()

    if batch_size is None:
        batch_size = model.training_generator.model.batch_size
    length_dataset = len(training_generator.dataset)
    assert batch_size <= length_dataset
    if path_file_name is not None:
        path_results = path_file_name[: path_file_name.rfind("/")]
        one_train_end_file = (
            path_results
            + "/"
            + path_file_name[path_file_name.rfind("/") + 1 :][0:-4]
            + "_final.txt"
        )
    else:
        one_train_end_file = None
        path_results = None
    missing_pred = False

    if epochs > 1:
        reset_logger = True
    else:
        reset_logger = False
    if one_train_end_file is None:
        reset_logger = False
    loggs = logger(
        model=model,
        train_generator=training_generator,
        test_generator=test_generator,
        on_epoch_end_file=path_file_name,
        on_train_end_file=one_train_end_file,
        device=device,
        lastlayer=not missing_pred,
        reset_logger=reset_logger,
        verbose=verbose,
    )
    if epochs == 0:
        return loggs

    # Generate variables and operations for the minimizer and initialize variables
    global_nelbow = np.inf
    train_per_epoch = len(training_generator.dataloader)

    ini_time = time.time()

    totla_time = 0.0

    num_divide = 1

    BEST_NELBO = np.inf
    optimizer = torch.optim.Adam(model.parameters(), lr=lrate, weight_decay=0.00)

    for el in range(num_divide):
        for param in model.parameters():
            param.requires_grad = True

        for e in range(epochs):
            message_write = {}
            message_write["epoch"] = e
            if verbose:
                kbar = pkbar.Kbar(
                    target=train_per_epoch,
                    epoch=e,
                    num_epochs=epochs,
                    width=50,
                    always_stateful=False,
                )

            avg_acc_c = 0
            avg_nll_c = 0

            start = time.time()

            idx = 0
            #if verbose:
            #    kbar.update(
            #        idx,
            #        values=[
            #            ("nelbo", 0),
                        #("rmse", 0),
                        #("nll", 0),
             #           ("rmse_c", 0),
             #           ("nll_c", 0),
             #       ],
             #   )
            avg_nelbo = 0.0
            avg_acc = 0
            avg_nll = 0

            idx += 1

            for i, data in enumerate(training_generator):
                batch_x, batch_y, batch_ind_nan, batch_ind_nan_target = data
                if batch_x.ndim == 3:
                    batch_x = np.transpose(batch_x, (1, 0, 2))

                loss = model.train_step(
                    optimizer,
                    batch_x.to(device),
                    batch_y.to(device),
                    batch_ind_nan,
                    batch_ind_nan_target,
                    e * train_per_epoch + i,
                    np.inf,
                )

                if i == 0:
                    y_aggr = model.y_aggr
                    mean_pred_aggr = model.mean_pred_aggr
                    std_pred_aggr = model.std_pred_aggr

                else:
                    if model.use_missing_gp:
                        y_aggr = torch.vstack((y_aggr, model.y_aggr))
                        mean_pred_aggr = torch.cat(
                            (mean_pred_aggr, model.mean_pred_aggr), 1
                        )
                        std_pred_aggr = torch.cat(
                            (std_pred_aggr, model.std_pred_aggr), 1
                        )
                    else:
                        y_aggr = torch.cat((y_aggr, model.y_aggr), 0)
                        mean_pred_aggr = torch.cat(
                            (mean_pred_aggr, model.mean_pred_aggr), 1
                        )
                        std_pred_aggr = torch.cat(
                            (std_pred_aggr, model.std_pred_aggr), 1
                        )
                acc = model.rmse_val
                nll = model.nll_val
                acc_c = model.rmse_val_converted
                nll_c = model.nll_val_converted
                avg_acc_c += acc_c
                avg_nll_c += nll_c
                avg_acc += acc
                avg_nll += nll
                avg_nelbo += loss
                if verbose:
                    kbar.update(
                        idx,
                        values=[
                            ("nelbo", loss),
                            #("rmse", acc),
                            #("nll", nll),
                            ("rmse_c", acc_c),
                            ("nll_c", nll_c),
                        ],
                    )
                idx += 1

            NELB, NLL, ACC, NLL_c, ACC_c = (
                avg_nelbo / train_per_epoch,
                avg_nll / train_per_epoch,
                avg_acc / train_per_epoch,
                avg_nll_c / train_per_epoch,
                avg_acc_c / train_per_epoch,
            )

            (
                message_write["acc_train"],
                message_write["nelbo_train"],
                message_write["nll_train"],
            ) = (
                ACC.detach().cpu().numpy().item(),
                NELB.detach().cpu().numpy().item(),
                NLL.detach().cpu().numpy().item(),
            )

            message_write["acc_train_c"], message_write["nll_train_c"] = (
                ACC_c.detach().cpu().numpy().item(),
                NLL_c.detach().cpu().numpy().item(),
            )

            if NELB < BEST_NELBO:
                BEST_NELBO = NELB
                if path_results is not None:
                    if model.use_missing_gp:
                        save_model(
                            model, optimizer, device, path_results, epoch="latest_missing"
                        )
                    else:
                        save_model(model, optimizer, device, path_results)

                    start_write_results = time.time()
                    ini_time = loggs.on_epoch_end(
                        ini_time, message_write, predict_test=predict_test
                    )

                    end_write_results = time.time()
                    write_time = end_write_results - start_write_results

                    if NELB < global_nelbow:
                        start_save_time = time.time()

                        end_save_time = time.time()
                        save_time = end_save_time - start_save_time
                        ini_time += save_time
                        global_nelbow = NELB

                        end = time.time() - save_time - write_time
                    else:
                        end = time.time() - write_time

                    totla_time += end - start
    """
    
    if model.use_missing_gp:
        load_model(
            model, optimizer, path_results, which_epoch="latest_missing", device=device
        )
    else:
        load_model(model, optimizer, path_results, which_epoch="latest", device=device)
    """
    # NELBO_train, NLL_train, ACC_train, NELBO_test, NLL_test, ACC_test, y_mst_train, y_mst_test = loggs.on_train_end()
    return loggs


def evaluate(
    model,
    training_generator,
    test_generator,
    path_file_name,
    loggs,
    batch_size=100,
    device=None,
):
    model.eval()

    #if batch_size is None:
    #    batch_size = model.training_generator.model.batch_size
    #length_dataset = len(test_generator.dataset)
    #assert batch_size <= length_dataset
    if path_file_name is not None:
        path_results = path_file_name[: path_file_name.rfind("/")]
        one_train_end_file = (
            path_results
            + "/"
            + path_file_name[path_file_name.rfind("/") + 1 :][0:-4]
            + "_final.txt"
        )
    else:
        path_results = None

    (
        NELBO_train,
        NLL_train,
        ACC_train,
        NELBO_test,
        NLL_test,
        ACC_test,
        train_info,
        test_info,
    ) = loggs.on_eval(training_generator, test_generator)

    return (
        NELBO_train,
        NLL_train,
        ACC_train,
        NELBO_test,
        NLL_test,
        ACC_test,
        train_info,
        test_info,
    )
