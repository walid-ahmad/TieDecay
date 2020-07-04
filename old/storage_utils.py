"""
Set of functions for saving and loading objects and matrices while
working with datasets

"""

import pickle
import numpy as np
from scipy import sparse

def save_obj(obj, filepath):
    """ Save object as a .pkl file. Flexible data formats """
    with open(filepath, 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(filepath):
    """ Load a .pkl file into memory """
    with open(filepath, 'rb') as f:
        return pickle.load(f)

def save_sparse_csr(filename,array):
    """ Efficient method for saving a sparse CSR matrix """
    np.savez(filename, data=array.data, indices=array.indices,
             indptr=array.indptr, shape=array.shape)

def load_sparse_csr(filename):
    """ Load a sparse CSR matrix """
    loader = np.load(filename)
    return sparse.csr_matrix((loader['data'], loader['indices'], loader['indptr']),
                         shape=loader['shape'])
