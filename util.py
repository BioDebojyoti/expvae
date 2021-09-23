import os
import sys
import requests

import argparse

import numpy as np
import torch
import torch.nn as nn

import pyro
import pyro.distributions as dist
from pyro.infer import SVI, JitTrace_ELBO, Trace_ELBO
from pyro.optim import Adam


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
