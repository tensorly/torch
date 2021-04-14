"""Tensor Regression Layers
"""

# Author: Jean Kossaifi
# License: BSD 3 clause

import torch
import torch.nn as nn

import tensorly as tl
tl.set_backend('pytorch')
from ..functional.tensor_regression import trl

from ..tensor_factorizations import FactorizedTensor

class TRL(nn.Module):
    """Base class for Tensor Regression Layers 
    
    Parameters
    -----------
    input_shape : int iterable
        shape of the input, excluding batch size
    output_shape : int iterable
        shape of the output, excluding batch size
    verbose : int, default is 0
        level of verbosity
    """
    def __init__(self, input_shape, output_shape, bias=False, verbose=0, 
                factorization='cp', rank='same', n_layers=1, **kwargs):
        super().__init__(**kwargs)
        self.verbose = verbose

        if isinstance(input_shape, int):
            self.input_shape = (input_shape, )
        else:
            self.input_shape = tuple(input_shape)
            
        if isinstance(output_shape, int):
            self.output_shape = (output_shape, )
        else:
            self.output_shape = tuple(output_shape)
        
        self.n_input = len(self.input_shape)
        self.n_output = len(self.output_shape)
        self.weight_shape = self.input_shape + self.output_shape
        self.order = len(self.weight_shape)

        if bias:
            self.bias = nn.Parameter(torch.Tensor(*self.output_shape))
        else:
            self.bias = None

        if n_layers == 1:
            factorization_shape = self.weight_shape
        elif isinstance(n_layers, int):
            factorization_shape = (n_layers, ) + self.weight_shape
        elif isinstance(n_layers, tuple):
            factorization_shape = n_layers + self.weight_shape
        
        if isinstance(factorization, FactorizedTensor):
            self.weight = factorization
        else:
            self.weight = FactorizedTensor.new(factorization_shape, rank=rank, factorization=factorization)

    def forward(self, x):
        """Performs a forward pass"""
        return trl(x, self.weight, bias=self.bias)
    
    def init_from_random(self, decompose_full_weight=False):
        """Initialize the module randomly

        Parameters
        ----------
        decompose_full_weight : bool, default is False
            if True, constructs a full weight tensor and decomposes it to initialize the factors
            otherwise, the factors are directly initialized randomlys        
        """
        if decompose_full_weight:
            full_weight = torch.normal(0.0, 0.02, size=self.weight_shape)
            self.weight.init_from_tensor(full_weight)
        else:
            self.weight.normal_()
        if self.bias is not None:
            self.bias.uniform_(-1, 1)
