# Missing Gaussian Processes

This code is a PyTorch 1.10.x cuda 11.3 implementation of the MGP method described in the paper entitled "Input Dependent Sparse Gaussian Processes", submitted to NeuroComputing. 
For comparision, we also implemented the following methods from :

DGP, VSGP and MGP



## Getting Started


### Prerequisites

The following package should be installed before using this code.

```
missingpy==0.2.0
numpy==1.19.5
pandas==1.1.5
pkbar==0.5
torch==1.10.0+cu113
matplotlib==3.3.0
fancyimpute==0.7.0
rpy2==3.4.5
scipy==1.5.0
scikit_learn==1.0.2

pip install -r requirements.txt

```

### Using the code
You can use the code as follows

```
put your data in "data" folder and run your experiments
you have the following optional arguments
python GP_experiment_torch.py -h
  -h, --help            show this help message and exit
  --dataset_name DATASET_NAME
                        name of the data set (should have subfolders with the
                        name s0, s1, s2, etc.) (default: None)
  --scaling SCALING     scaling method [MeanStd|MinMax|MaxAbs|Robust|None]
                        (default: MeanStd)
  --split_number split_number
                        data set split number [0|1|2|etc] (default: 0)
  --name svgp
                        the available SVGP models are [nn|solve|swsgp|titsias]
                        (default: nn)
  --Ptype PTYPE         Problem type (regression or classification)
                        [reg|class] (default: reg)
  --nGPU NGPU           GPU number (for cpu use -1) [-1|0|1|2] (default: -1)
  --nEpoch NEPOCH       Maximum number of epochs (default: 100) (default: 100)
  --minibatch_size BATCHSIZE
                        Batch size (default: 100) (default: 100)
  --M NIP             number of inducing points (default: 100) (default:
                        1024)
  --M2 NIP2             number of inducing points (default: 100) (default:
                        1024)
  --M2 NIP2             number of inducing points (default: 100) (default:
                        1024)     
  --imputation mean     mean|median|knn|mice|None
  
  --kernel              Matern|RBF
  
  --likelihood_var      ariance noise gaussian likelihood
  
  --lrate               learning rate
  
  --missing             consider missing rate
  
  --nGPU                GPU number
  
  --n_epoch             number of training epochs
  
  --n_samples           number of MC samples
  
  --nolayers            number of layers
  
  --numThreads          number of threads
  
  --var_noise           variance noise
  
  --consider_miss       consider missing for DGP and VSGP

```

You can run experiments ucing UCI data set with the above options.



## Cite
@article{jafrasteh2022gaussian,
  title={Gaussian Processes for Missing Value Imputation},
  author={Jafrasteh, Bahram and Hern{\'a}ndez-Lobato, Daniel and Lubi{\'a}n-L{\'o}pez, Sim{\'o}n Pedro and Benavente-Fern{\'a}ndez, Isabel},
  journal={arXiv preprint arXiv:2204.04648},
  year={2022}
}
[link to the paper](https://arxiv.org/pdf/2204.04648.pdf)


## License

This project is licensed under the MIT License.

