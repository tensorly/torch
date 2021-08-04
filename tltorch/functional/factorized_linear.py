import torch
import tensorly as tl
from collections import Counter
from tensorly.tt_tensor import TTTensor

tl.set_backend('pytorch')

# Author: Taylor Lee Patti <taylorpatti@g.harvard.edu>

def tt_factorized_linear(tt_vec, ttm_weights):
    """Contracts a TT tensor with a TT matrix and returns a TT tensor.

    Parameters
    ----------
    tt_vec : tensor train tensor
    ttm_weights : tensor train matrix

    Returns
    -------
    The tensor train tensor obtained for contracting the TT tensor and the TT matrix.
    """
    ncores = len(tt_vec)
    contr_layer = []
    for i in range(ncores):
        dimW, dimX = ttm_weights[i].shape, tt_vec[i].shape
        contr = tl.einsum('abc,debf->adecf', tt_vec[i], ttm_weights[i])
        contr_layer.append(tl.reshape(contr, (dimW[0]*dimX[0], dimW[1], dimW[3]*dimX[2])))
    return TTTensor(contr_layer)
