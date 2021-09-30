import sys
import scanpy as sc
import pandas as pd
import random


'''
    This function splits the gex(scRNA) and peaks(scATAC) AnnData into test and train sets
'''


def train_test(gex,peaks,train_percent=0.70):

    samples_indices = gex.obs.index.tolist()

    training_set_samples = random.sample(samples_indices,int(train_percent*float(len(samples_indices))))
    test_set_samples = [s for s in samples_indices if s not in training_set_samples]

    train_gex = gex[gex.obs.index.isin(training_set_samples)].copy()
    train_peaks = peaks[peaks.obs.index.isin(training_set_samples)].copy()

    test_gex = gex[gex.obs.index.isin(test_set_samples)].copy()
    test_peaks = peaks[peaks.obs.index.isin(test_set_samples)].copy()



    return train_gex, train_peaks, test_gex, test_peaks


