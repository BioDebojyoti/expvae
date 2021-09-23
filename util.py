import os
import sys
import requests
import argparse
import numpy as np
import torch
import torch.nn as nn

import scvi

import pyro
import pyro.distributions as dist
from pyro.infer import SVI, JitTrace_ELBO, Trace_ELBO
from pyro.optim import Adam


import anndata
from torch.utils.data import DataLoader
from typing import Optional, Union



# helps loading tensors from AnnData objects 
class AnnDataLoader(DataLoader):
    """
    DataLoader for loading tensors from AnnData objects.

    Parameters
    ----------
    adata
        An anndata objects
    shuffle
        Whether the data should be shuffled
    indices
        The indices of the observations in the adata to load
    batch_size
        minibatch size to load each iteration
    data_and_attributes
        Dictionary with keys representing keys in data registry (`adata.uns["_scvi"]`)
        and value equal to desired numpy loading type (later made into torch tensor).
        If `None`, defaults to all registered data.
    data_loader_kwargs
        Keyword arguments for :class:`~torch.utils.data.DataLoader`
    """

    def __init__(
        self,
        adata: anndata.AnnData,
        shuffle=False,
        indices=None,
        batch_size=128,
        data_and_attributes: Optional[dict] = None,
        drop_last: Union[bool, int] = False,
        **data_loader_kwargs,
    ):

        if "_scvi" not in adata.uns.keys():
            raise ValueError("Please run setup_anndata() on your anndata object first.")

        if data_and_attributes is not None:
            data_registry = adata.uns["_scvi"]["data_registry"]
            for key in data_and_attributes.keys():
                if key not in data_registry.keys():
                    raise ValueError(
                        "{} required for model but not included when setup_anndata was run".format(
                            key
                        )
                    )

        self.dataset = AnnTorchDataset(adata, getitem_tensors=data_and_attributes)

        sampler_kwargs = {
            "batch_size": batch_size,
            "shuffle": shuffle,
            "drop_last": drop_last,
        }

        if indices is None:
            indices = np.arange(len(self.dataset))
            sampler_kwargs["indices"] = indices
        else:
            if hasattr(indices, "dtype") and indices.dtype is np.dtype("bool"):
                indices = np.where(indices)[0].ravel()
            indices = np.asarray(indices)
            sampler_kwargs["indices"] = indices

        self.indices = indices
        self.sampler_kwargs = sampler_kwargs
        sampler = BatchSampler(**self.sampler_kwargs)
        self.data_loader_kwargs = copy.copy(data_loader_kwargs)
        # do not touch batch size here, sampler gives batched indices
        self.data_loader_kwargs.update({"sampler": sampler, "batch_size": None})

        super().__init__(self.dataset, **self.data_loader_kwargs)

# define the PyTorch module that parameterizes the
# diagonal gaussian distribution q(z|x)
class Encoder(nn.Module):
    def __init__(self, z_dim, hidden_dim):
        super().__init__()
        # setup the three linear transformations used
        # TODO change the encoder to encode count matrix instead of image !!
        # rows == genes, columns == samples
        #self.fc1 = nn.Linear(784, hidden_dim)
        #self.fc21 = nn.Linear(hidden_dim, z_dim)
        #self.fc22 = nn.Linear(hidden_dim, z_dim)
        ## setup the non-linearities
        #self.softplus = nn.Softplus()

    def forward(self, x):
        # define the forward computation on the image x
        # first shape the mini-batch to have pixels in the rightmost dimension
        x = x.reshape(-1, 784)
        # then compute the hidden units
        hidden = self.softplus(self.fc1(x))
        # then return a mean vector and a (positive) square root covariance
        # each of size batch_size x z_dim
        z_loc = self.fc21(hidden)
        z_scale = torch.exp(self.fc22(hidden))
        return z_loc, z_scale

class dataset:

   def download(self, url):
       target_path = url.split("/")[-1]
       print(url)
       print(target_path)
       response = requests.get(url, stream=True)
       if response.status_code == 200:
           with open(target_path, 'wb') as f:
               f.write(response.raw.read())

#if __name__ == "__main__":
#    
#    url = "http://cf.10xgenomics.com/samples/cell-exp/1.1.0/pbmc3k/pbmc3k_filtered_gene_bc_matrices.tar.gz"
#    dataset_instance = dataset()
#    dataset_instance.download(url)
#
