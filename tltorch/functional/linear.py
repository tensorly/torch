import numpy as np
import torch
import torch.nn.functional as F
from ..factorized_tensors import TensorizedTensor
from ..factorized_tensors.tensorized_matrices import CPTensorized, TuckerTensorized, BlockTT
from .factorized_linear import linear_blocktt, linear_cp, linear_tucker
import pdb

import tensorly as tl
tl.set_backend('pytorch')

# Author: Jean Kossaifi
# License: BSD 3 clause


def factorized_linear(x, weight, bias=None, in_features=None):
    """Linear layer with a dense input x and factorized weight
    """
    if in_features is None:
        in_features = np.prod(x.shape[-1])

    if not torch.is_tensor(weight):
        # Weights are in the form (out_features, in_features) 
        # PyTorch's linear returns dot(x, weight.T)!
        if isinstance(weight, TensorizedTensor):
            x_shape = x.shape[:-1] + weight.tensorized_shape[1]
            out_shape = x.shape[:-1] + (-1, )
            if isinstance(weight, CPTensorized):
                if bias is None:
                    return linear_cp(x.reshape(x_shape), weight).reshape(out_shape)
                else:
                    return linear_cp(x.reshape(x_shape), weight).reshape(out_shape) + bias
            elif isinstance(weight, TuckerTensorized):
                if bias is None:
                    return linear_tucker(x.reshape(x_shape), weight).reshape(out_shape)
                else:
                    return linear_tucker(x.reshape(x_shape), weight).reshape(out_shape) + bias
            elif isinstance(weight, BlockTT):
                if bias is None:
                    return linear_blocktt(x.reshape(x_shape), weight).reshape(out_shape)
                else:
                    return linear_blocktt(x.reshape(x_shape), weight).reshape(out_shape) + bias
            # if no efficient implementation available: use reconstruction
            weight = weight.to_matrix()
        else:
            weight = weight.to_tensor()
        
    return F.linear(x, torch.reshape(weight, (-1, in_features)), bias=bias)
