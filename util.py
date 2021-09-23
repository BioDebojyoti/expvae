import os
import sys
import requests


class dataset:

   def download(self, url):
       target_path = url.split("/")[-1]
       print(url)
       print(target_path)
       response = requests.get(url, stream=True)
       if response.status_code == 200:
           with open(target_path, 'wb') as f:
               f.write(response.raw.read())

if __name__ == "__main__":
    
    url = "http://cf.10xgenomics.com/samples/cell-exp/1.1.0/pbmc3k/pbmc3k_filtered_gene_bc_matrices.tar.gz"
    dataset_instance = dataset()
    dataset_instance.download(url)

