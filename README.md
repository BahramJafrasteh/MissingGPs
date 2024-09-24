# Missing Gaussian Processes

This code is a PyTorch 1.10.x cuda 11.3 implementation of the MGP method described in the paper entitled "Input Dependent Sparse Gaussian Processes". 
For comparison, we also implemented the following methods from:

DGP, VSGP and MGP

## Getting Started

### Prerequisites

Install the dependencies using the following command:

```bash
pip install -r requirements.txt
```

### Using the code
You can use the code as follows

```
put your data in "datasets" folder and run your experiments
you have the following optional arguments
python run_experiment.py -h
  -h, --help            show this help message and exit
  --dataset_name DATASET_NAME
                        name of the data set (should have subfolders with the
                        name s0, s1, s2, etc.) (default: None)
  --scaling SCALING     scaling method [MeanStd|MinMax|MaxAbs|Robust|None]
                        (default: MeanStd)
  --split_number split_number
                        data set split number [0|1|2|etc] (default: 0)
  --name svgp           svgp
  --nGPU NGPU           GPU number (for cpu use -1) [-1|0|1|2] (default: -1)
  --minibatch_size BATCHSIZE
                        Batch size (default: 100) (default: 100)
  --M NIP             number of inducing points (default: 100) (default:
                        1024)
  --M2 NIP2             number of inducing points (default: 100) (default:
                        1024)
  --imputation mean     mean|median|knn|mice|None
  
  --kernel              Matern|RBF (defaults:matern)
  
  --likelihood_var      ariance noise gaussian likelihood (0.01)
  
  --lrate               learning rate (0.01)
  
  --missing             consider missing (should be on for MGP, otherwise return normal SVGP)
  
  --nGPU                GPU number
  
  --n_epoch             number of training epochs
  
  --n_samples           number of MC samples
  
  --nolayers            number of layers
  
  --numThreads          number of threads
  
  --var_noise           variance noise
  
  --consider_miss       consider missing for DGP and VSGP

```

You can run experiments ucing UCI data set with the above options.
To replicate results from the paper:
python run_experiment.py --dataset_name parkinson_10 --lrate 0.01 --split_number 0 --name svgp --n_samples 20 --M 100 --M2 100 --no_iterations 10000 --nolayers 1 --nGPU 0 --minibatch_size 100 --fitting --imputation mean --missing

## Cite
Jafrasteh, B., Hernández-Lobato, D., Lubián-López, S. P., & Benavente-Fernández, I. (2023). Gaussian processes for missing value imputation. Knowledge-Based Systems, 273, 110603.
[Missing GPs](https://www.sciencedirect.com/science/article/pii/S0950705123003532)

## License

This project is licensed under the MIT License.

