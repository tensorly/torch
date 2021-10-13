import numpy as np
import torch
import torch.nn.functional as F
from ..factorized_tensors import TensorizedTensor

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
            weight = weight.to_matrix()
        else:
            weight = weight.to_tensor()


    return F.linear(x, torch.reshape(weight, (-1, in_features)), bias=bias)
