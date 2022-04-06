
import torch
from inducing_layer_dgp.torch.data.dataloader import dataloader
from inducing_layer_dgp.torch.data.dataloader_single import dataloader as dataloader_single


class customLoader(object):
    def __init__(self, X, y, ind_total_nan, ind_total_nan_target, minibatch_size,  numThreads= 1, shuffle = True, device=None):

        self.dataset = dataloader()
        self.dataset.create(X, y, ind_total_nan, ind_total_nan_target, device=device)
        # load data set
        self.dataloader = torch.utils.data.DataLoader(
            self.dataset,
            batch_size=minibatch_size,
            shuffle=shuffle,
           # num_workers=numThreads
        )

        self.initialize()
    def name(self):
        return 'customLoader'

    def initialize(self):
        pass

    def __len__(self):
        return len(self.dataset)

    def __iter__(self):
        for i, data in enumerate(self.dataloader):
            yield data