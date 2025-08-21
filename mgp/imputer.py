import torch
import numpy as np
import warnings
import os
import random

from torch.nn import ModuleList
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Assuming the following MGP modules are in the python path
# You might need to adjust the imports based on your project structure
from mgp.model.data import customLoader
from mgp.model.kernels import SquaredExponential as RBF
from mgp.model.kernels import White, Matern
from mgp.model.likelihoods import Gaussian
from mgp.model.train import train, evaluate
from mgp.model.dgp import DGP
from mgp.model.callback import logger
from mgp.model.layers import SVGPLayer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import SimpleImputer
from sklearn.impute import IterativeImputer
from sklearn.linear_model import LinearRegression, BayesianRidge
from sklearn.preprocessing import RobustScaler
from sklearn.impute import KNNImputer

# Set a seed for reproducibility
def set_seed(seed):
    """Sets the seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.cuda.manual_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

class MGPImputer(BaseEstimator, TransformerMixin):
    """
    Missing Value Imputation using Deep Gaussian Processes.

    This class wraps the MGP implementation into a scikit-learn compatible
    estimator, allowing for easy integration into ML pipelines.

    Parameters
    ----------
    n_layers : int, default=1
        Number of layers in the Deep GP.
    kernel : str, default='matern'
        Kernel function to use ('rbf' or 'matern').
    imputation_strategy : str, default='chained'
        Strategy for imputation. Can be 'holistic' or 'chained'.
        'holistic': Builds a single multi-output DGP for all features.
        'chained': Builds a separate GP layer for each feature with missing values. (MGP)
    n_inducing_points : int, default=100
        Number of inducing points (M) for the sparse GP.
    n_samples : int, default=10
        Number of samples to draw for Monte Carlo estimation.
    learning_rate : float, default=0.01
        Learning rate for the Adam optimizer.
    n_iterations : int, default=2000
        Number of training iterations.
    batch_size : int, default=128
        Size of minibatches for training.
    var_noise : float, default=0.1
        Initial variance for the White kernel (noise).
    likelihood_var : float, default=0.1
        Initial variance for the Gaussian likelihood.
    use_cuda : bool, default=True
        Whether to use GPU if available.
    seed : int, default=0
        Random seed for reproducibility.
    """
    def __init__(self, n_layers=2, kernel='matern', n_inducing_points=100,imputation_strategy='chained',
                 n_samples=20, learning_rate=0.01, n_iterations=10000,
                 batch_size=128, var_noise=0.0001, likelihood_var=0.01,imp_init='mean',
                 use_cuda=True, verbose=True, seed=0):

        self.n_layers = n_layers
        self.kernel = kernel
        self.n_inducing_points = n_inducing_points
        self.n_samples = n_samples
        self.imputation_strategy = imputation_strategy
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.batch_size = batch_size
        self.var_noise = var_noise
        self.likelihood_var = likelihood_var
        self.use_cuda = use_cuda
        self.seed = seed
        self.imputation_method = imp_init
        # Internal attributes
        self.model_ = None
        self.scaler_ = StandardScaler()
        self.device_ = None
        self.layer_missing_index_ = []
        self.verbose=verbose


    def _initialize_device(self):
        """Initializes the torch device (CPU or CUDA)."""
        if self.use_cuda and torch.cuda.is_available():
            self.device_ = torch.device("cuda")
            torch.cuda.set_device(0) # Assuming first GPU
            torch.backends.cudnn.benchmark = True
        else:
            self.device_ = torch.device("cpu")
        print(f"Using device: {self.device_.type}")

    def _build_holistic_model(self, X_train, y_train, ind_nan):
        """Constructs a single, multi-output DGP model for all features."""
        # Select kernel
        kernel_used = Matern if self.kernel == "matern" else RBF

        Z, lengthscale = self._get_inducing_points_and_lengthscale(X_train)

        # Construct the network layers
        D = X_train.shape[1]
        layer_sizes = []
        kernels = ModuleList()
        likelihood_gaussian = ModuleList()

        for _ in range(self.n_layers):
            kernels.append(
                kernel_used(lengthscales=np.tile(lengthscale, D), variance=100.0, device=self.device_) +
                White(variance=self.var_noise, device=self.device_)
            )
            layer_sizes.append(D)

        num_data = [X_train.shape[0]]
        likelihood_gaussian.append(Gaussian(self.likelihood_var, device=self.device_))

        # Convert data to torch tensors and create the DGP model
        X_train_torch = torch.from_numpy(X_train).to(torch.float64).to(self.device_)
        y_train_torch = torch.from_numpy(y_train).to(torch.float64).to(self.device_)
        Z_torch = torch.from_numpy(Z).to(torch.float64).to(self.device_)

        return DGP(
            X=X_train_torch, Y=y_train_torch, Z=Z_torch, kernels=kernels,
            layer_sizes=layer_sizes, likelihood_gaussian=likelihood_gaussian,
            likelihood_ber=None,
            num_data=np.array(num_data), num_samples=self.n_samples,
            y_mean=torch.from_numpy(self.x_means).to(self.device_, torch.float64),
            y_std=torch.from_numpy(self.x_stds).to(self.device_, torch.float64),
            device=self.device_, ind_total_nan=ind_nan,
            layer_missing_index=self.layer_missing_index_,
            x_means=self.x_means, x_stds=self.x_stds,
            layer=SVGPLayer, use_missing_gp=False
        )

    def _get_inducing_points_and_lengthscale(self, X_train):
        """Helper to calculate inducing points and initial lengthscale."""
        # Create a copy without NaNs for KMeans and distance calculation
        X_train_nonan = np.nan_to_num(X_train, nan=np.nanmean(X_train, axis=0))

        # Determine inducing points using KMeans
        try:
            kmeans = KMeans(n_clusters=self.n_inducing_points, random_state=self.seed, n_init=10).fit(X_train_nonan)
            Z = kmeans.cluster_centers_
        except Exception as e:
            if self.n_inducing_points>= X_train.shape[0]:
                raise exit(f'The number of inducing points {self.n_inducing_points} should be less than data rows {X_train.shape[0]}')
            print(f"KMeans failed: {e}. Falling back to random sampling for inducing points.")
            indices = np.random.choice(X_train.shape[0], self.n_inducing_points, replace=False)
            Z = X_train_nonan[indices, :]

        # Estimate initial lengthscale based on median distance
        if X_train.shape[0] < 1000:
            value = np.expand_dims(np.sum(np.square(X_train_nonan), 1), 1)
            distance = value - 2 * np.matmul(X_train_nonan, np.transpose(X_train_nonan)) + np.transpose(value)
            median = np.percentile(distance, q=50)
        else:
            median = 10.0
        lengthscale = np.log(median) if median > 0 else 1.0

        return Z, lengthscale

    def _build_chained_model(self, X_train, y_train, ind_nan):
        """Constructs a DGP with a separate layer for each feature with missing data."""
        # Select kernel
        kernel_used = Matern if self.kernel == "matern" else RBF

        Z, lengthscale = self._get_inducing_points_and_lengthscale(X_train)

        # Identify and order features with missing values
        total_mis = ind_nan.sum(0)
        features_with_missing = np.where(total_mis > 0)[0]

        # Sort by number of missing values (descending)
        sorted_indices_by_missing = np.argsort(-total_mis[features_with_missing])
        layers_missing_ordered = features_with_missing[sorted_indices_by_missing]

        # Then sort by standard deviation (ascending)
        stds_of_missing_layers = self.x_stds[layers_missing_ordered]
        sorted_indices_by_std = np.argsort(stds_of_missing_layers)
        self.layer_missing_index_ = layers_missing_ordered[sorted_indices_by_std]

        print(f"Chained model processing features in order: {self.layer_missing_index_}")

        num_data = []
        layer_sizes = []
        likelihood_gaussian = ModuleList()
        kernels = torch.nn.ModuleList()

        for layer_idx in self.layer_missing_index_:
            # Each layer predicts one feature from all others
            nD = X_train.shape[1] - 1
            num_data.append((~ind_nan[:, layer_idx]).sum())
            kernels.append(
                kernel_used(lengthscales=np.tile(lengthscale, nD), variance=1.0, device=self.device_) +
                White(variance=self.var_noise, device=self.device_)
            )
            layer_sizes.append(nD)
            likelihood_gaussian.append(Gaussian(self.likelihood_var, device=self.device_))

        if not self.layer_missing_index_.any():
            raise ValueError("Chained strategy selected, but no missing data found.")

        # Convert data to torch tensors and create the DGP model
        X_train_torch = torch.from_numpy(X_train).to(torch.float64).to(self.device_)
        y_train_torch = torch.from_numpy(y_train).to(torch.float64).to(self.device_)
        Z_torch = torch.from_numpy(Z).to(torch.float64).to(self.device_)

        return DGP(
            X=X_train_torch, Y=y_train_torch, Z=Z_torch, kernels=kernels,
            layer_sizes=layer_sizes, likelihood_gaussian=likelihood_gaussian,
            likelihood_ber=None,
            num_data=np.array(num_data), num_samples=self.n_samples,
            y_mean=torch.from_numpy(self.x_means).to(self.device_, torch.float64),
            y_std=torch.from_numpy(self.x_stds).to(self.device_, torch.float64),
            device=self.device_, ind_total_nan=ind_nan,
            layer_missing_index=self.layer_missing_index_,
            x_means=self.x_means, x_stds=self.x_stds,
            layer=SVGPLayer, use_missing_gp=True
        )


    def initial_imputer(self, X):
        if self.imputation_method == 'median':
            imp = SimpleImputer(missing_values=np.nan, strategy='median')
        elif self.imputation_method == 'mean':
            imp = SimpleImputer(missing_values=np.nan, strategy='mean')
        elif self.imputation_method == 'constant':
            imp = SimpleImputer(missing_values=np.nan, strategy='constant', fill_value=0.0)
        elif self.imputation_method == 'mice':
            #lr = BayesianRidge()
            lr = LinearRegression()
            imp = IterativeImputer(estimator=lr, missing_values=np.nan, max_iter=10, verbose=-1,
                                   random_state=42)
        elif self.imputation_method == 'knn':
            imp = KNNImputer(n_neighbors=20, add_indicator=False)
        else:
            raise Exception('Please see the instruction...')
        X = imp.fit_transform(X).astype('float')
        self.init_imputer = imp
        return X

    def compute_mean_std(self, X_train, ind_used_train):
        means, stds = [], []

        for i in range(X_train.shape[1]):
            std = X_train[~ind_used_train[:, i], i].std()
            if std != 0:
                means.append(X_train[~ind_used_train[:, i], i].mean())

                stds.append(std)
            else:
                means.append(0)

                stds.append(0)
        return np.array(means), np.array(stds)

    def fit(self, X, X_preimputed=None):
        """
        Fit the MGP imputer on the data.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Input data with missing values, where missing values are encoded as np.nan.

        Returns
        -------
        self : object
            Returns self.
        """
        set_seed(self.seed)
        self._initialize_device()

        # 1. Preprocessing
        X_train = X.copy()
        ind_nan = np.isnan(X_train)
        
        # Fit scaler on the data (ignoring NaNs for mean/std calculation)
        #self.scaler_.fit(X_train)

        #X_scaled = self.scaler_.transform(X_train)
        
        # Replace NaNs with 0 after scaling, as the model expects numerical input.
        # The `ind_nan` mask will tell the model which values were originally missing.
        #X_imputed_mean = np.nan_to_num(X_scaled, nan=0.0)
        if X_preimputed is None:
            X_imputed_mean = self.initial_imputer(X_train)
        else:
            X_imputed_mean = X_preimputed
        x_means, x_stds = self.compute_mean_std(X_imputed_mean, ind_nan)

        self.x_means = (np.median(X_imputed_mean, 0) + np.mean(X_imputed_mean, 0)) / 2.0
        self.x_stds = X_imputed_mean.std(0) + 0.00001

        X_imputed_mean = (X_imputed_mean - self.x_means) / self.x_stds

        # 2. Build Model
        print(f"Building model with '{self.imputation_strategy}' strategy...")
        if self.imputation_strategy == 'holistic':
            self.model_ = self._build_holistic_model(X_imputed_mean, X_imputed_mean, ind_nan)
        elif self.imputation_strategy == 'chained':
            self.model_ = self._build_chained_model(X_imputed_mean, X_imputed_mean, ind_nan)
        else:
            raise ValueError(f"Unknown imputation_strategy: {self.imputation_strategy}")
        
        params = list(self.model_.parameters())
        print(f"Model has {np.sum([p.numel() for p in params])} trainable parameters.")
        
        optimizer = torch.optim.Adam(self.model_.parameters(), lr=self.learning_rate)

        # 3. Training
        # The customLoader needs a "full" dataset for its loss calculation.
        # Since we are doing imputation, the "full" data is the mean-imputed data.

        X_train_full = X_imputed_mean.copy()

        
        # The model expects samples as the first dimension
        X_train_samples = np.tile(np.expand_dims(X_imputed_mean, 0), [self.n_samples, 1, 1])

        training_generator = customLoader(
            X_train_samples,
            X_train_full,
            ind_nan,
            ind_nan,
            self.batch_size,
            shuffle=True,
            device=self.device_,
        )



        n_data_points = X_train.shape[0]
        iter_per_epoch = max(1, n_data_points / self.batch_size)
        epochs = int(np.ceil(self.n_iterations / iter_per_epoch))

        print(f"Starting training for {epochs} epochs...")
        train(
            model=self.model_,
            training_generator=training_generator,
            test_generator=None, # No test set during fitting
            optimizer=optimizer,
            path_file_name=None, # Don't save to file
            batch_size=self.batch_size,
            epochs=epochs,
            device=self.device_,
            layer_missing_index=self.layer_missing_index_,
            lrate=self.learning_rate,
            predict_test=False,
            verbose=self.verbose,
        )
        print("Training complete.")
        return self


    def predict(self, X, batch_size=None, X_preimputed=None, return_prob=False):
        """
        Impute missing values in X.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Data with missing values to impute.

        Returns
        -------
        X_imputed : array-like, shape (n_samples, n_features)
            Data with missing values imputed.
        """
        if batch_size is None:
            batch_size = self.batch_size
        if self.model_ is None:
            raise RuntimeError("You must call fit before calling predict.")

        X_test = X.copy()
        ind_nan_test = np.isnan(X_test)

        if X_preimputed is None:
            X_imputed_test = self.init_imputer.transform(X_test)
        else:
            X_imputed_test = X_preimputed

        X_imputed_test = (X_imputed_test - self.x_means) / self.x_stds
        # Scale data using the scaler fitted on the training data
        #X_scaled = self.scaler_.transform(X_test)
        #X_imputed_mean = np.nan_to_num(X_scaled, nan=0.0)
        #X_test_full = X_imputed_mean.copy()

        X_test_samples = np.tile(np.expand_dims(X_imputed_test, 0), [self.n_samples, 1, 1])

        # Create a data loader for the test set
        test_generator = customLoader(
            X_test_samples,
            X_imputed_test,
            ind_nan_test,
            ind_nan_test,
            batch_size,
            shuffle=False,
            device=self.device_,
        )
        loggs = logger(
            model=self.model_,
            train_generator=None,
            test_generator=test_generator,
            on_epoch_end_file=None,
            on_train_end_file=None,
            device=self.device_,
            lastlayer=True,
            reset_logger=False,
            verbose=self.verbose,
        )
        # Use the evaluate function to get predictions
        # We only need the test_info part of the output
        _, _, _, _, _, _, _, test_info = evaluate(
            model=self.model_,

            training_generator=None,
            test_generator=test_generator,
            path_file_name=None,
            loggs=loggs,
            batch_size=batch_size,
            device=self.device_,
        )

        _, _, mean_pred, var_pred_aggr_test, ind_nan_aggr_test = test_info

        # Inverse transform to get back to the original data scale
        var_pred_aggr_test = (var_pred_aggr_test * (self.x_stds[None, None, :]) ** 2)
        std_pred_test = np.sqrt(var_pred_aggr_test + 0.0001)
        mean_pred = mean_pred*self.x_stds[None, None, :] + self.x_means[None, None, :]



        # The model returns predictions for all points. We only care about the missing ones.
        # We take the mean of the samples from the posterior.
        mean_imputation_scaled = np.mean(mean_pred, axis=0)
        std_imputatino_scaled = np.mean(std_pred_test, axis=0)
        # Create the final standard deviation
        X_std = np.zeros_like(std_imputatino_scaled)
        X_std[ind_nan_test] = std_imputatino_scaled[ind_nan_test]
        # Create the final imputed dataset
        X_imputed = X.copy()
        X_imputed[ind_nan_test] = mean_imputation_scaled[ind_nan_test]

        if return_prob:
            return X_imputed, X_std, mean_pred, std_pred_test
        else:
            return X_imputed, X_std

    def fit_transform(self, X, X_preimputed=None, **fit_params):
        """
        Fit to data, then transform it.
        This is a convenience method that fits the model and then returns the imputed data.
        """
        return self.fit(X, X_preimputed,**fit_params).predict(X, X_preimputed=X_preimputed)


if __name__ == '__main__':
    # Example Usage
    print("Running MGP Imputer Example...")
    import pandas as pd
    # 1. Create a synthetic dataset with missing values
    np.random.seed(42)

    #n_samples, n_features = 200, 5
    #X_true = np.random.rand(n_samples, n_features) * 10
    #X_missing = X_true.copy()
    # Introduce 20% missing values
    #missing_mask = np.random.rand(n_samples, n_features) < 0.2
    #X_missing[missing_mask] = np.nan

    #print(f"Created a dataset of shape {X_missing.shape} with {np.sum(missing_mask)} missing values.")


    X_true = pd.read_csv('datasets/parkinson.csv').values
    X_missing = pd.read_csv('datasets/parkinson_10.csv').values
    n_samples, n_features = X_true.shape
    missing_mask = np.isnan(X_missing.astype(np.float64)) > 0


    # 2. Initialize and fit the imputer
    # Using fewer iterations for a quick example run
    mgp_imputer = MGPImputer(
        n_layers=1,
        n_inducing_points=100,
        n_iterations=500,
        learning_rate=0.01,
        batch_size=100,
        imputation_strategy='holistic',#holistic or chained
        imp_init='mean',
        use_cuda=True,
        seed=42
    )

    # 3. Fit on the data with missing values and then transform it
    X_imputed, X_std_imputed = mgp_imputer.fit_transform(X_missing, X_preimputed=None)

    # 4. Evaluate the imputation
    rmse = np.sqrt(np.mean((X_imputed[missing_mask] - X_true[missing_mask])**2))
    print(f"\nImputation complete.")
    print(f"RMSE on missing values: {rmse:.4f}")

    # You can now use the `mgp_imputer` to predict on new data
    # For example, create a new test set with missing values
    #X_test_true = np.random.rand(50, n_features) * 10
    #X_test_missing = X_test_true.copy()
    #test_missing_mask = np.random.rand(50, n_features) < 0.2
    #X_test_missing[test_missing_mask] = np.nan

    X_test_true = pd.read_csv('datasets/parkinson_test.csv').values
    X_test_missing = pd.read_csv('datasets/parkinson_test_10.csv').values
    test_missing_mask = np.isnan(X_test_missing.astype(np.float64)) > 0


    X_test_imputed, X_std_imputed = mgp_imputer.predict(X_test_missing)
    test_rmse = np.sqrt(np.mean((X_test_imputed[test_missing_mask] - X_test_true[test_missing_mask])**2))
    print(f"RMSE on a new test set: {test_rmse:.4f}")

