import csv
from tabnanny import verbose

import torch
import time
import pkbar
from mgp.model.utils import *
import pandas as pd
from os.path import join, dirname
import numpy as np


class logger:
    def __init__(
        self,
        model,
        train_generator,
        test_generator,
        on_epoch_end_file,
        on_train_end_file,
        device=None,
        lastlayer=False,
        reset_logger=True,
        verbose=False,
    ):
        self.on_epoch_end_file = on_epoch_end_file
        self.on_train_end_file = on_train_end_file
        self.train_generator = train_generator
        self.test_generator = test_generator
        self.total_train_time = 0.0
        self.model = model
        self.verbose = verbose
        self.lastlayer = lastlayer
        self.device = device
        if reset_logger:
            with open(on_epoch_end_file, "w") as myfile:
                log_writer = csv.writer(myfile)
                log_writer.writerow(
                    [
                        "epoch",
                        "nelbo_train",
                        "acc_train",
                        "nll_train",
                        "acc_train_c",
                        "nll_train_c",
                        "nelbo_test",
                        "acc_test",
                        "nll_test",
                        "acc_test_c",
                        "nll_test_c",
                        "total_time_train",
                        "prediction_time",
                    ]
                )
            with open(on_train_end_file, "w") as myfile:
                log_writer = csv.writer(myfile)
                log_writer.writerow(
                    [
                        "RMSE_train",
                        "NLL_train",
                        "RMSE_train_c",
                        "NLL_train_c",
                        "RMSE_test",
                        "NLL_test",
                        "RMSE_test_c",
                        "NLL_test_c",
                        "total_training_time",
                        "prediction_time",
                    ]
                )

    def evaluate(self, generator, normalize_y=True):
        with torch.no_grad():
            _per_epoch = len(generator.dataloader)
            if self.verbose:
                kbar = pkbar.Kbar(target=_per_epoch, width=50, always_stateful=False)
                kbar.update(
                    0,
                    values=[
                        ("nelbo", 0),
                        #("rmse", 0),
                        #("nll", 0),
                        ("rmse_c", 0),
                        ("nll_c", 0),
                    ],
                )
            # idx += 1
            avg_nelbo = 0
            avg_acc = 0
            avg_nll = 0
            avg_acc_c = 0
            avg_nll_c = 0
            for idx, data in enumerate(generator):
                batch_x, batch_y, batch_ind_nan, batch_ind_nan_target = data
                if batch_x.ndim == 3:
                    batch_x = np.transpose(batch_x, (1, 0, 2))

                loss = self.model.test_step(
                    batch_x.to(self.device),
                    batch_y.to(self.device),
                    batch_ind_nan,
                    batch_ind_nan_target,
                    normalize_y=True,
                )

                if idx == 0:
                    y_aggr = self.model.y_aggr
                    x_aggr = batch_x
                    mean_pred_aggr = self.model.mean_pred_aggr
                    std_pred_aggr = self.model.std_pred_aggr
                    ind_nan_aggr = batch_ind_nan

                else:
                    if self.model.use_missing_gp:
                        y_aggr = torch.vstack((y_aggr, self.model.y_aggr))
                        x_aggr = torch.cat((x_aggr, batch_x), 1)
                        mean_pred_aggr = torch.cat(
                            (mean_pred_aggr, self.model.mean_pred_aggr), 1
                        )
                        std_pred_aggr = torch.cat(
                            (std_pred_aggr, self.model.std_pred_aggr), 1
                        )
                        ind_nan_aggr = torch.cat((ind_nan_aggr, batch_ind_nan), 0)

                    else:
                        y_aggr = torch.vstack((y_aggr, self.model.y_aggr))
                        x_aggr = torch.cat((x_aggr, batch_x), 1)
                        mean_pred_aggr = torch.cat(
                            (mean_pred_aggr, self.model.mean_pred_aggr), 1
                        )
                        std_pred_aggr = torch.cat(
                            (std_pred_aggr, self.model.std_pred_aggr), 1
                        )
                        ind_nan_aggr = torch.cat((ind_nan_aggr, batch_ind_nan), 0)

                avg_nelbo += loss.detach().cpu().numpy()
                acc = self.model.rmse_val
                nll = self.model.nll_val
                acc_c = self.model.rmse_val_converted
                nll_c = self.model.nll_val_converted
                avg_acc += acc.detach().cpu().numpy()
                avg_nll += nll.detach().cpu().numpy()
                avg_acc_c += acc_c.detach().cpu().numpy()
                avg_nll_c += nll_c.detach().cpu().numpy()
                if self.verbose:
                    kbar.update(
                        idx + 1,
                        values=[
                            ("nelbo", loss),
                            #("rmse", acc),
                            #("nll", nll),
                            ("rmse_c", acc_c),
                            ("nll_c", nll_c),
                        ],
                    )
                # idx += 1
        NLL, ACC = compute_nn_mse(y_aggr, mean_pred_aggr, std_pred_aggr, ind_nan_aggr)
        NLL = NLL.detach().cpu().numpy()
        ACC = ACC.detach().cpu().numpy()
        NLL_c, ACC_c = compute_nn_mse(
            y_aggr * self.model.y_std + self.model.y_mean,
            mean_pred_aggr * self.model.y_std + self.model.y_mean,
            std_pred_aggr * self.model.y_std,
            ind_nan_aggr,
        )
        NLL_c = NLL_c.detach().cpu().numpy()
        ACC_c = ACC_c.detach().cpu().numpy()
        #print(ACC, NLL, ACC_c, NLL_c)
        #print("\n")

        y_m_std = torch.stack((mean_pred_aggr, std_pred_aggr)).transpose(0, 1)

        NELBO, _, _ = avg_nelbo / _per_epoch, avg_nll / _per_epoch, avg_acc / _per_epoch

        return (
            NELBO,
            NLL,
            ACC,
            NLL_c,
            ACC_c,
            [x_aggr, y_aggr, mean_pred_aggr, std_pred_aggr, ind_nan_aggr],
        )

    def on_epoch_end(self, ini_time, message_write={}, predict_test=True):
        """
        write the results to the file the firs columns is RMSE, scond one is Standaridzed RMSE, third one NLL, last one time
        @param means: computed posterior means
        @param variances: computed posterior variances
        @param output: target points
        @param ini_time: time start
        @param file_name: output file path and name
        @return: nothing
        """
        start_predict = time.time()
        NELBO, NLL, ACC, NLL_c, ACC_c = 0.0, 0.0, 0.0, 0.0, 0.0
        self.model.eval()

        if predict_test:
            NELBO, NLL, ACC, NLL_c, ACC_c, _ = self.evaluate(
                self.test_generator, normalize_y=False
            )
        end_predict = time.time()
        prediction_time = end_predict - start_predict
        ini_time += prediction_time
        print("\n")
        self.total_train_time = time.time() - ini_time
        with open(self.on_epoch_end_file, "a") as myfile:
            log_writer = csv.writer(myfile)
            log_writer.writerow(
                [
                    message_write["epoch"],
                    message_write["nelbo_train"],
                    message_write["acc_train"],
                    message_write["nll_train"],
                    message_write["acc_train_c"],
                    message_write["nll_train_c"],
                    NELBO,
                    ACC,
                    NLL,
                    ACC_c,
                    NLL_c,
                    self.total_train_time,
                    prediction_time,
                ]
            )

        self.model.train()

        return ini_time

    def on_train_end(self, training_gen=None, test_gen=None):
        pass

    def on_eval(self, training_gen=None, test_gen=None):
        start_predict = time.time()
        self.model.eval()

        if training_gen:
            self.train_generator = training_gen
        if test_gen:
            self.test_generator = test_gen

        if not self.model.use_missing_gp:
            end_predict = time.time()
            (
                NELBO_test,
                NLL_test,
                ACC_test,
                NLL_test_c,
                ACC_test_c,
                [
                    x_aggr_test,
                    y_aggr_test,
                    mean_pred_aggr_test,
                    std_pred_aggr_test,
                    ind_nan_aggr_test,
                ],
            ) = self.evaluate(self.test_generator, normalize_y=False)

            x_aggr_test = x_aggr_test.detach().cpu().numpy()
            y_aggr_test = y_aggr_test.detach().cpu().numpy()
            mean_pred_aggr_test = mean_pred_aggr_test.detach().cpu().numpy()
            std_pred_aggr_test = std_pred_aggr_test.detach().cpu().numpy()
            ind_nan_aggr_test = ind_nan_aggr_test.detach().cpu().numpy()
            print("\n")
            if self.train_generator is not None:
                (
                    NELBO_train,
                    NLL_train,
                    ACC_train,
                    NLL_train_c,
                    ACC_train_c,
                    [
                        x_aggr_train,
                        y_aggr_train,
                        mean_pred_aggr_train,
                        std_pred_aggr_train,
                        ind_nan_aggr_train,
                    ],
                ) = self.evaluate(self.train_generator, normalize_y=False)

                if torch.is_tensor(mean_pred_aggr_train):
                    y_aggr_train = y_aggr_train.detach().cpu().numpy()
                    x_aggr_train = x_aggr_train.detach().cpu().numpy()
                    mean_pred_aggr_train = mean_pred_aggr_train.detach().cpu().numpy()
                    std_pred_aggr_train = std_pred_aggr_train.detach().cpu().numpy()
                    ind_nan_aggr_train = ind_nan_aggr_train.detach().cpu().numpy()


            else:
                NELBO_train = None,
                NLL_train = None,
                ACC_train = None,
                NLL_train_c = None,
                ACC_train_c = None,
                x_aggr_train = None,
                y_aggr_train = None,
                mean_pred_aggr_train = None,
                std_pred_aggr_train = None,
                ind_nan_aggr_train = None,


        else:
            end_predict = time.time()
            if self.train_generator is not None:
                (
                    NELBO_train,
                    NLL_train,
                    ACC_train,
                    NLL_train_c,
                    ACC_train_c,
                    [
                        x_aggr_train,
                        y_aggr_train,
                        mean_pred_aggr_train,
                        std_pred_aggr_train,
                        ind_nan_aggr_train,
                    ],
                ) = self.evaluate(self.train_generator, normalize_y=False)
                if torch.is_tensor(mean_pred_aggr_train):
                    y_aggr_train = y_aggr_train.detach().cpu().numpy()
                    x_aggr_train = x_aggr_train.detach().cpu().numpy()
                    mean_pred_aggr_train = mean_pred_aggr_train.detach().cpu().numpy()
                    std_pred_aggr_train = std_pred_aggr_train.detach().cpu().numpy()
                    ind_nan_aggr_train = ind_nan_aggr_train.detach().cpu().numpy()

            else:
                NELBO_train = None,
                NLL_train = None,
                ACC_train=None,
                NLL_train_c=None,
                ACC_train_c=None,
                x_aggr_train = None,
                y_aggr_train = None,
                mean_pred_aggr_train = None,
                std_pred_aggr_train = None,
                ind_nan_aggr_train = None,
            (
                NELBO_test,
                NLL_test,
                ACC_test,
                NLL_test_c,
                ACC_test_c,
                [
                    x_aggr_test,
                    y_aggr_test,
                    mean_pred_aggr_test,
                    std_pred_aggr_test,
                    ind_nan_aggr_test,
                ],
            ) = self.evaluate(self.test_generator, normalize_y=False)
            if torch.is_tensor(mean_pred_aggr_test):
                x_aggr_test = x_aggr_test.detach().cpu().numpy()
                y_aggr_test = y_aggr_test.detach().cpu().numpy()
                mean_pred_aggr_test = mean_pred_aggr_test.detach().cpu().numpy()
                std_pred_aggr_test = std_pred_aggr_test.detach().cpu().numpy()
                ind_nan_aggr_test = ind_nan_aggr_test.detach().cpu().numpy()


        NELBO_test = 0.0
        prediction_time = end_predict - start_predict
        """
        
        if self.verbose:
            print("Average likelihood {:.4f}".format(ACC_test))
            print("Average rmse {:.4f}".format(NLL_test))
            print(
                "prediction time {}".format( prediction_time
                )
            )
        """
        #outfile = join(dirname(self.on_train_end_file), "results_missing_pred_test.csv")
        if self.on_train_end_file is not None:
            with open(self.on_train_end_file, "a") as myfile:
                log_writer = csv.writer(myfile)
                log_writer.writerow(
                    [
                        ACC_train,
                        NLL_train,
                        ACC_train_c,
                        NLL_train_c,
                        ACC_test,
                        NLL_test,
                        ACC_test_c,
                        NLL_test_c,
                        self.total_train_time,
                        prediction_time,
                    ]
                )
        return (
            NELBO_train,
            NLL_train,
            ACC_train,
            NELBO_test,
            NLL_test,
            ACC_test,
            [
                x_aggr_train,
                y_aggr_train,
                mean_pred_aggr_train,
                std_pred_aggr_train,
                ind_nan_aggr_train,
            ],
            [
                x_aggr_test,
                y_aggr_test,
                mean_pred_aggr_test,
                std_pred_aggr_test,
                ind_nan_aggr_test,
            ],
        )
