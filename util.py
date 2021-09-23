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
from typing import Optional, Union, List

# prepares anndata.AnnData for conversion to tensor
class preprocessanndata2tensor(anndata.AnnData):
    def __init__(self):
        super().__init__()

    def setup_anndata(self,
        adata: anndata.AnnData,
        batch_key: Optional[str] = None,
        labels_key: Optional[str] = None,
        layer: Optional[str] = None,
        protein_expression_obsm_key: Optional[str] = None,
        protein_names_uns_key: Optional[str] = None,
        categorical_covariate_keys: Optional[List[str]] = None,
        continuous_covariate_keys: Optional[List[str]] = None,
        copy: bool = False,
    ) -> Optional[anndata.AnnData]:
        """
        Sets up :class:`~anndata.AnnData` object for `scvi` models.

        A mapping will be created between data fields used by `scvi` to their respective locations in adata.
        This method will also compute the log mean and log variance per batch for the library size prior.

        None of the data in adata are modified. Only adds fields to adata.

        Parameters
        ----------
        adata
            AnnData object containing raw counts. Rows represent cells, columns represent features.
        batch_key
            key in `adata.obs` for batch information. Categories will automatically be converted into integer
            categories and saved to `adata.obs['_scvi_batch']`. If `None`, assigns the same batch to all the data.
        labels_key
            key in `adata.obs` for label information. Categories will automatically be converted into integer
            categories and saved to `adata.obs['_scvi_labels']`. If `None`, assigns the same label to all the data.
        layer
            if not `None`, uses this as the key in `adata.layers` for raw count data.
        protein_expression_obsm_key
            key in `adata.obsm` for protein expression data, Required for :class:`~scvi.model.TOTALVI`.
        protein_names_uns_key
            key in `adata.uns` for protein names. If None, will use the column names of `adata.obsm[protein_expression_obsm_key]`
            if it is a DataFrame, else will assign sequential names to proteins. Only relevant but not required for :class:`~scvi.model.TOTALVI`.
        categorical_covariate_keys
            keys in `adata.obs` that correspond to categorical data. Used in some `scvi` models.
        continuous_covariate_keys
            keys in `adata.obs` that correspond to continuous data. Used in some `scvi` models.
        copy
            if `True`, a copy of adata is returned.

        Returns
        -------
        If ``copy``,  will return :class:`~anndata.AnnData`.
        Adds the following fields to adata:

        .uns['_scvi']
            `scvi` setup dictionary
        .obs['_local_l_mean']
            per batch library size mean
        .obs['_local_l_var']
            per batch library size variance
        .obs['_scvi_labels']
            labels encoded as integers
        .obs['_scvi_batch']
            batch encoded as integers

        Examples
        --------
        Example setting up a scanpy dataset with random gene data and no batch nor label information

        >>> import scanpy as sc
        >>> import scvi
        >>> import numpy as np
        >>> adata = scvi.data.synthetic_iid(run_setup_anndata=False)
        >>> adata
        AnnData object with n_obs × n_vars = 400 × 100
            obs: 'batch', 'labels'
            uns: 'protein_names'
            obsm: 'protein_expression'

        Filter cells and run preprocessing before `setup_anndata`

        >>> sc.pp.filter_cells(adata, min_counts = 0)

        Since no batch_key nor labels_key was passed, setup_anndata() will assume all cells have the same batch and label

        >>> scvi.data.setup_anndata(adata)
        INFO      No batch_key inputted, assuming all cells are same batch
        INFO      No label_key inputted, assuming all cells have same label
        INFO      Using data from adata.X
        INFO      Computing library size prior per batch
        INFO      Registered keys:['X', 'batch_indices', 'local_l_mean', 'local_l_var', 'labels']
        INFO      Successfully registered anndata object containing 400 cells, 100 vars, 1 batches, 1 labels, and 0 proteins. Also registered 0 extra categorical covariates and 0 extra continuous covariates.

        Example setting up scanpy dataset with random gene data, batch, and protein expression

        >>> adata = scvi.data.synthetic_iid(run_setup_anndata=False)
        >>> scvi.data.setup_anndata(adata, batch_key='batch', protein_expression_obsm_key='protein_expression')
        INFO      Using batches from adata.obs["batch"]
        INFO      No label_key inputted, assuming all cells have same label
        INFO      Using data from adata.X
        INFO      Computing library size prior per batch
        INFO      Using protein expression from adata.obsm['protein_expression']
        INFO      Generating sequential protein names
        INFO      Registered keys:['X', 'batch_indices', 'local_l_mean', 'local_l_var', 'labels', 'protein_expression']
        INFO      Successfully registered anndata object containing 400 cells, 100 vars, 2 batches, 1 labels, and 100 proteins. Also registered 0 extra categorical covariates and 0 extra continuous covariates.
        """
        if copy:
            adata = adata.copy()

        if adata.is_view:
            raise ValueError(
                "Please run `adata = adata.copy()` or use the copy option in this function."
            )

        adata.uns["_scvi"] = {}
        adata.uns["_scvi"]["scvi_version"] = scvi.__version__

        batch_key = _setup_batch(adata, batch_key)
        labels_key = _setup_labels(adata, labels_key)
        x_loc, x_key = _setup_x(adata, layer)
        local_l_mean_key, local_l_var_key = _setup_library_size(adata, batch_key, layer)

        data_registry = {
            _CONSTANTS.X_KEY: {"attr_name": x_loc, "attr_key": x_key},
            _CONSTANTS.BATCH_KEY: {"attr_name": "obs", "attr_key": batch_key},
            _CONSTANTS.LOCAL_L_MEAN_KEY: {"attr_name": "obs", "attr_key": local_l_mean_key},
            _CONSTANTS.LOCAL_L_VAR_KEY: {"attr_name": "obs", "attr_key": local_l_var_key},
            _CONSTANTS.LABELS_KEY: {"attr_name": "obs", "attr_key": labels_key},
        }

        if protein_expression_obsm_key is not None:
            protein_expression_obsm_key = _setup_protein_expression(
                adata, protein_expression_obsm_key, protein_names_uns_key, batch_key
            )
            data_registry[_CONSTANTS.PROTEIN_EXP_KEY] = {
                "attr_name": "obsm",
                "attr_key": protein_expression_obsm_key,
            }

        if categorical_covariate_keys is not None:
            cat_loc, cat_key = _setup_extra_categorical_covs(
                adata, categorical_covariate_keys
            )
            data_registry[_CONSTANTS.CAT_COVS_KEY] = {
                "attr_name": cat_loc,
                "attr_key": cat_key,
            }

        if continuous_covariate_keys is not None:
            cont_loc, cont_key = _setup_extra_continuous_covs(
                adata, continuous_covariate_keys
            )
            data_registry[_CONSTANTS.CONT_COVS_KEY] = {
                "attr_name": cont_loc,
                "attr_key": cont_key,
            }

        # add the data_registry to anndata
        _register_anndata(adata, data_registry_dict=data_registry)
        logger.debug("Registered keys:{}".format(list(data_registry.keys())))
        _setup_summary_stats(
            adata,
            batch_key,
            labels_key,
            protein_expression_obsm_key,
            categorical_covariate_keys,
            continuous_covariate_keys,
        )

        logger.info("Please do not further modify adata until model is trained.")

        _verify_and_correct_data_format(adata, data_registry)

        if copy:
            return adata


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
