import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
#from fancyimpute import KNN
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.linear_model import LinearRegression, BayesianRidge
from sklearn.preprocessing import (
    MinMaxScaler,
    MaxAbsScaler,
    StandardScaler,
    RobustScaler,
)
from sklearn.impute import KNNImputer
from sklearn.metrics import mean_squared_error
#from rpy2.robjects.packages import importr
#import rpy2.robjects as ro
#import rpy2.robjects.numpy2ri

#rpy2.robjects.numpy2ri.activate()


def mean_imputer(x, ind_total_nan):
    for o in range(x.shape[1]):
        ind_true = ~ind_total_nan[:, o]
        uq = np.mean(x[ind_true, o])
        x[ind_total_nan[:, o], :][:, o] = uq

    return x


def load_dataset(opt):
    ######################## inputs ###############################

    dta = opt.dataset_name.split("_")[0]  #'clinicaldata'

    df_train_missing = pd.read_csv(
        "datasets/{}/s{}/{}_train_data.csv".format(
            opt.dataset_name, opt.split_number, dta
        )
    )
    df_train_lables = pd.read_csv(
        "datasets/{}/s{}/{}_train_labels.csv".format(
            opt.dataset_name, opt.split_number, dta
        ),
    )

    df_train_ = pd.concat((df_train_missing, df_train_lables), axis=1)
    df_add_missing = pd.read_csv(
        "datasets/{}/s{}/{}_test_data.csv".format(
            opt.dataset_name, opt.split_number, dta
        ),
    )
    df_test_labels = pd.read_csv(
        "datasets/{}/s{}/{}_test_labels.csv".format(
            opt.dataset_name, opt.split_number, dta
        ),
    )
    df_test_ = pd.concat((df_add_missing, df_test_labels), axis=1)

    df_train_full = pd.read_csv(
        "datasets/{}/s{}/{}_train_data.csv".format(dta, opt.split_number, dta)
    )
    df_train_full_l = pd.read_csv(
        "datasets/{}/s{}/{}_train_labels.csv".format(dta, opt.split_number, dta)
    )
    df_train_full = pd.concat((df_train_full, df_train_full_l), axis=1)

    df_test_full = pd.read_csv(
        "datasets/{}/s{}/{}_test_data.csv".format(dta, opt.split_number, dta),
    )
    df_test_full_l = pd.read_csv(
        "datasets/{}/s{}/{}_test_labels.csv".format(dta, opt.split_number, dta),
    )
    df_test_full = pd.concat((df_test_full, df_test_full_l), axis=1)

    if "id" in df_train_.columns:
        df_train = df_train_.drop("id", axis=1)
        df_test = df_test_.drop("id", axis=1)
    else:
        df_train = df_train_
        df_test = df_test_

    df_current_train = df_train
    df_current_test = df_test

    global ind_binary
    ind_binary = []
    r = 0
    for col in df_current_train:
        uq = [u for u in df_current_train[col].unique() if u == u]
        if len(uq) == 2:
            ind_binary.append(r)
        r += 1
    ind_binary = []
    cols = df_current_train.columns

    if opt.nolayers == np.inf:
        X_test = df_current_test.values.astype("float")
        X_train = df_current_train.values.astype("float")
        y_train = None
        y_test = None
    else:
        df_current_train = df_current_train.replace("?", np.nan)
        df_current_test = df_current_test.replace("?", np.nan)

        X_train = df_current_train.values.astype("float")
        X_test = df_current_test.values.astype("float")
        y_train = df_current_train[cols[-1]].values.astype("float")
        y_test = df_current_test[cols[-1]].values.astype("float")

        df_train_full = df_train_full.replace("?", np.nan)
        df_test_full = df_test_full.replace("?", np.nan)
        X_train_full = df_train_full.values.astype("float")
        X_test_full = df_test_full.values.astype("float")

    if opt.imputation == "median":
        imp = SimpleImputer(missing_values=np.nan, strategy="median")
    elif opt.imputation == "mean":
        imp = SimpleImputer(missing_values=np.nan, strategy="mean")
    elif opt.imputation == "constant":
        imp = SimpleImputer(missing_values=np.nan, strategy="constant", fill_value=0.0)
    elif opt.imputation == "mice":
        lr = LinearRegression()
        imp = IterativeImputer(
            estimator=lr, missing_values=np.nan, max_iter=10, verbose=2, random_state=0
        )
    elif opt.imputation == "linear":
        lr = LinearRegression()
        imp = IterativeImputer(
            estimator=lr, missing_values=np.nan, max_iter=1, verbose=2, random_state=0
        )

    elif opt.imputation == "knn":
        imp = KNNImputer(n_neighbors=2, add_indicator=False)
    else:
        raise Exception("Please see the instruction...")

    if opt.nolayers == np.inf:
        ind_n_test = np.isnan(X_test)
        for ir in range(ind_n_test.shape[1]):
            indnan = ind_n_test[:, ir]
            ind_true = ~indnan
            uq = np.unique(X_test[ind_true, ir])
            if indnan.sum() > 0:
                if len(uq) <= 4:
                    X_test[indnan, ir] = np.mean(uq)
                    X_train[indnan, ir] = np.mean(uq)
                else:
                    X_test[indnan, ir] = np.mean(X_test[ind_true, ir])
                    X_train[indnan, ir] = np.mean(X_test[ind_true, ir])

    if opt.nolayers == np.inf:
        ind_total_nan = np.isnan(df_current_train.values.astype(np.float64)) > 0
        ind_total_nan = ind_total_nan * ~ind_n_test
        rmse_inint = np.sqrt(
            mean_squared_error(X_train[ind_total_nan], X_test[ind_total_nan])
        )
        print("rMSE INIT {}, {}".format(opt.imputation, rmse_inint))
    else:
        ind_total_nan = np.isnan(df_current_train.values.astype(np.float64)) > 0

    global ind_total_nan_test

    ind_total_nan_test = np.isnan(df_current_test.values.astype(np.float64)) > 0
    ind_used_train = df_current_train.isna().values
    if opt.imputation != "rmean":
        X_train = imp.fit_transform(X_train).astype("float")
        X_test = imp.transform(X_test).astype("float")
        X_test_full = imp.transform(X_test_full).astype("float")
        X_train_full = imp.transform(X_train_full).astype("float")

    else:
        X_train = mean_imputer(X_train, ind_total_nan)

    x_means, x_stds = scaler_fit(X_train, ind_used_train)
    if opt.nolayers == np.inf:
        X_train = scaler_transform(x_means, x_stds, X_train)
        X_test = scaler_transform(x_means, x_stds, X_test)
        scaler = None
    else:
        x_means = (np.median(X_train, 0) + np.mean(X_train, 0)) / 2.0
        x_stds = X_train.std(0) + 0.00001
        rmse = []
        for d in range(ind_total_nan_test.shape[1]):
            if ind_total_nan_test[:, d].sum() > 0:
                a = np.sqrt(
                    np.mean(
                        (
                            X_test[ind_total_nan_test[:, d], d]
                            - X_test_full[ind_total_nan_test[:, d], d]
                        )
                        ** 2
                    )
                )
                rmse.append(a)
        rmse = np.stack(rmse)
        print("mean imputation: ", np.mean(rmse))

        X_train = (X_train - x_means) / x_stds
        X_test = (X_test - x_means) / x_stds
        X_train_full = (X_train_full - x_means) / x_stds
        X_test_full = (X_test_full - x_means) / x_stds
        rmse = []
        for d in range(ind_total_nan_test.shape[1]):
            if ind_total_nan_test[:, d].sum() > 0:
                a = np.sqrt(
                    np.mean(
                        (
                            X_test[ind_total_nan_test[:, d], d]
                            - X_test_full[ind_total_nan_test[:, d], d]
                        )
                        ** 2
                    )
                )
                rmse.append(a)
        rmse = np.stack(rmse)
        print("mean imputation: ", np.mean(rmse))

        scaler = None  # RobustScaler()

    if opt.nolayers != np.inf:
        y_mean = np.mean(y_train).reshape(1)
        y_std = np.std(y_train).reshape(1)
        y_train = (y_train - y_mean) / y_std
        y_test = (y_test - y_mean) / y_std
    else:
        y_mean, y_std, y_train, y_test = None, None, None, None
    return [
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
    ]


def scaler_fit(X_train, ind_used_train):
    means, stds = [], []

    for i in range(X_train.shape[1]):
        std = X_train[~ind_used_train[:, i], i].std()
        if std != 0:
            means.append(X_train[~ind_used_train[:, i], i].mean())

            stds.append(std)
        else:
            means.append(0)

            stds.append(0)
    return means, stds


def scaler_transform(means, stds, X_train):
    for i in range(X_train.shape[1]):
        if stds[i] != 0:
            X_train[:, i] = (X_train[:, i] - means[i]) / stds[i]

    return X_train


def prepare_data(df_current_train, df_current_test, target_ind="VCT_"):
    print("preprocessing data ...")
    for el in df_current_train:
        if el == "id":
            continue
        if df_current_train[el].dtype == "O":
            list_uqs = list(
                set(
                    np.hstack(
                        [df_current_test[el].unique(), df_current_train[el].unique()]
                    )
                )
            )
            uq = [u for u in list_uqs if u == u]
            imputed_values = [0, 1]
            uq = np.sort(uq)
            if len(uq) == 2:
                # df_current_train[el] = df_current_train[el].astype('int')
                len_train = df_current_train.shape[0]
                dummies = pd.get_dummies(
                    pd.concat([df_current_train[el], df_current_test[el]], 0),
                    drop_first=True,
                ).astype("object")
                dummies.columns = [
                    el + "_" + col for col in list(dummies.columns.astype("str"))
                ]
                ind_n = (
                    pd.concat([df_current_train[el], df_current_test[el]], 0)
                    .isna()
                    .values
                )
                for col in dummies.columns:
                    dummies[col][dummies[col] == 0] = 0
                    dummies[col][dummies[col] == 1] = 1
                    dummies[col][ind_n] = np.nan
                df_current_train = pd.concat([df_current_train, dummies[:len_train]], 1)
                df_current_test = pd.concat([df_current_test, dummies[len_train:]], 1)
                df_current_train = df_current_train.drop(el, 1)
                df_current_test = df_current_test.drop(el, 1)
                ls = list(df_current_train.columns)
                ls.remove("VCT_")
                ls.append("VCT_")
                df_current_train = df_current_train[ls]
                df_current_test = df_current_test[ls]

    return df_current_train, df_current_test  # , means, stds
