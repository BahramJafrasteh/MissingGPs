# MGP-Imputer: Missing Value Imputation with Deep Gaussian Processes
[![PyPI version](https://badge.fury.io/py/mgp-imputer.svg)](https://badge.fury.io/py/mgp-imputer)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A PyTorch-based implementation of Missing Gaussian Processes (MGP) for missing value imputation, wrapped in a user-friendly `scikit-learn` compatible API.

This package allows you to seamlessly integrate Deep Gaussian Process models into your data preprocessing pipelines for robust and uncertainty-aware imputation. It is based on the paper ["Gaussian processes for missing value imputation"](https://www.sciencedirect.com/science/article/pii/S0950705123003532).

<p align="center">
  <img src="https://raw.githubusercontent.com/BahramJafrasteh/MissingGPs/main/mgp.png" alt="MELAGE Demo" width="600"/><br>
  <p align="center">
    <em>Chained Deep Gaussian Processes for Missing Value Imputation</em>  
  </p>
</p>

## Features

- **Scikit-learn Compatible:** Use `fit`, `predict`, and `fit_transform` methods just like any other scikit-learn transformer.
- **Two Imputation Strategies:**
    - `chained` (Default): Builds a separate GP layer for each feature with missing values, modeling dependencies in a chained fashion (MGP).
    - `holistic`: Builds a single, multi-output Deep GP to model all features simultaneously.
- **Probabilistic Imputation:** Returns both the imputed values and the standard deviation, giving you a measure of uncertainty for each imputed value.
- **GPU Accelerated:** Leverages PyTorch to run on CUDA devices for significant speedups.

## Installation

You can install `mgp-imputer` directly from PyPI:

```bash
pip install mgp-imputer
```



## **Quick Start**
Here's how to use `MGPImputer` to fill in missing values (`np.nan`) in your dataset.
```bash
import numpy as np
import pandas as pd
from mgp import MGPImputer

# 1. Create a synthetic dataset with 20% missing values
np.random.seed(42)
n_samples, n_features = 200, 5
X_true = np.random.rand(n_samples, n_features) * 10
X_missing = X_true.copy()
missing_mask = np.random.rand(n_samples, n_features) < 0.2
X_missing[missing_mask] = np.nan

print(f"Created a dataset with {np.sum(missing_mask)} missing values.")

# 2. Initialize the MGPImputer
# Strategies can be 'chained' (default) or 'holistic'
imputer = MGPImputer(
    imputation_strategy='chained',
    n_inducing_points=100,
    n_iterations=1000, # Use more iterations for real data
    learning_rate=0.01,
    batch_size=64,
    verbose=True,
    seed=42
)

# 3. Fit on the data and transform it to get imputed values
# The imputer returns the imputed data and the standard deviation of the predictions
X_imputed, X_std = imputer.fit_transform(X_missing)

# 4. Evaluate the imputation quality
rmse = np.sqrt(np.mean((X_imputed[missing_mask] - X_true[missing_mask])**2))
print(f"\nImputation complete.")
print(f"RMSE on missing values: {rmse:.4f}")

# The result is a complete numpy array
print("\nImputed data shape:", X_imputed.shape)
print("Number of NaNs in imputed data:", np.isnan(X_imputed).sum())
```

## Configuration Options

You can customize the behavior of `MGPImputer` by passing parameters during initialization. Here are the available options:

| Parameter | Description | Type | Options | Default |
|---|---|---|---|---|
| `imputation_strategy` | The core method for building the GP model. | `str` | `'chained'`, `'holistic'` | `'chained'` |
| `imp_init` | The standard imputation method used to create an initial complete dataset before training the GP. | `str` | `'mean'`, `'median'`, `'knn'`, `'mice'`, `'constant'` | `'mean'` |
| `kernel` | The covariance function for the Gaussian Process layers. | `str` | `'matern'`, `'rbf'` | `'matern'` |
| `n_layers` | The number of layers in the Deep GP. Only used when `imputation_strategy` is `'holistic'`. | `int` | `> 0` | `2` |
| `n_inducing_points` | The number of inducing points for the sparse GP approximation. A higher number is more accurate but computationally slower. | `int` | `> 0` | `100` |
| `n_iterations` | The total number of optimization iterations to run during training. | `int` | `> 0` | `10000` |
| `n_samples` | The number of Monte Carlo samples drawn to approximate the model's posterior distribution. | `int` | `> 0` | `20` |
| `learning_rate` | The learning rate for the Adam optimizer. | `float` | `> 0` | `0.01` |
| `batch_size` | The number of data points in each mini-batch during training. | `int` | `> 0` | `128` |
| `likelihood_var` | The initial variance of the Gaussian likelihood function. | `float` | `> 0` | `0.01` |
| `var_noise` | The initial variance of the white noise added to the kernel. | `float` | `> 0` | `0.0001` |
| `verbose` | If `True`, prints training progress and other information. | `bool` | `True`, `False` | `True` |
| `use_cuda` | If `True`, the model will run on a GPU if one is available. | `bool` | `True`, `False` | `True` |
| `seed` | A random seed for reproducibility of results. | `int` | `≥ 0` | `0` |

## **Citation**
If you use this work in your research, please cite the original paper:

Jafrasteh, B., Hernández-Lobato, D., Lubián-López, S. P., & Benavente-Fernández, I. (2023). Gaussian processes for missing value imputation. Knowledge-Based Systems, 273, 110603.
[Missing GPs](https://www.sciencedirect.com/science/article/pii/S0950705123003532)



## License

This project is licensed under the MIT License.

