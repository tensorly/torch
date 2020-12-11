"""Module for initializing tensor decompositions
"""

# Author: Jean Kossaifi
# License: BSD 3 clause

import torch
import math
import numpy as np
import tensorly as tl
tl.set_backend('pytorch')


def cp_init(weights, factors, std=0.02):
    """Initializes directly the weights and factors of a CP decomposition so the reconstruction has the specified std and 0 mean

    Parameters
    ----------
    weights : 1D tensor
    factors : list of 2D factors of size (dim_i, rank)
    std : float, default is 0.02
        the desired standard deviation of the full (reconstructed) tensor

    Notes
    -----
    We assume the given (weights, factors) form a correct CP decomposition, no checks are done here.
    """
    rank = factors[0].shape[1] # We assume we are given a valid CP 
    order = len(factors)
    std_factors = (std/math.sqrt(rank))**(1/order)

    with torch.no_grad():
        weights.fill_(1)
        for i in range(len(factors)):
            factors[i].normal_(0, std_factors)
    return weights, factors

def tucker_init(core, factors, std=0.02):
    """Initializes directly the weights and factors of a Tucker decomposition so the reconstruction has the specified std and 0 mean

    Parameters
    ----------
    weights : 1D tensor
    factors : list of 2D factors of size (dim_i, rank)
    std : float, default is 0.02
        the desired standard deviation of the full (reconstructed) tensor
    
    Notes
    -----
    We assume the given (core, factors) form a correct Tucker decomposition, no checks are done here.
    """
    order = len(factors)
    rank = tl.shape(core)
    r = np.prod([math.sqrt(r) for r in rank])
    std_factors = (std/r)**(1/(order+1))
    with torch.no_grad():
        core.normal_(0, std_factors)
        for i in range(len(factors)):
            factors[i].normal_(0, std_factors)
    return core, factors

def tt_init(factors, std=0.02):
    """Initializes directly the weights and factors of a TT decomposition so the reconstruction has the specified std and 0 mean

    Parameters
    ----------
    weights : 1D tensor
    factors : list of 2D factors of size (dim_i, rank)
    std : float, default is 0.02
        the desired standard deviation of the full (reconstructed) tensor
    
    Notes
    -----
    We assume the given factors form a correct TT decomposition, no checks are done here.
    """
    order = len(factors)
    r = np.prod([math.sqrt(f.shape[2]) for f in factors[:-1]])
    std_factors = (std/r)**(1/order)
    with torch.no_grad():
        for i in range(len(factors)):
            factors[i].normal_(0, std_factors)
    return factors

def tt_matrix_init(factors, std=0.02):
    """Initializes directly the weights and factors of a TT-Matrix decomposition so the reconstruction has the specified std and 0 mean

    Parameters
    ----------
    weights : 1D tensor
    factors : list of 2D factors of size (dim_i, rank)
    std : float, default is 0.02
        the desired standard deviation of the full (reconstructed) tensor
    
    Notes
    -----
    We assume the given factors form a correct TT-Matrix decomposition, no checks are done here.
    """
    return tt_init(factors, std=std)