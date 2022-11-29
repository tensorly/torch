
import math

import numpy as np
import torch
from torch import nn

import tensorly as tl
tl.set_backend('pytorch')
from tensorly import tenalg
from tensorly.decomposition import tucker, tensor_train

from tltorch.factorized_tensors.core import FactorizedTensor
from tltorch.factorized_tensors.factorized_tensors import TuckerTensor, CPTensor, TTTensor, DenseTensor
from tltorch.utils import FactorList


# Author: Jean Kossaifi
# License: BSD 3 clause

class ComplexHandler():
    _complex_params = []
    def __setattr__(self, key, value):
        if key in self._complex_params:
            if isinstance(value, FactorList):
                value = FactorList([nn.Parameter(torch.view_as_real(f)) if isinstance(f, nn.Parameter) else torch.view_as_real(f)\
                                     for f in value])
            elif isinstance(value, nn.Parameter):
                value = nn.Parameter(torch.view_as_real(value))
            else:
                value = torch.view_as_real(value)
            key = '_' + key
        super().__setattr__(key, value)
    
    def __getattr__(self, key):
        if key in self._complex_params:
            value = getattr(self, "_" + key)
            if isinstance(value, FactorList):
                return FactorList([torch.view_as_complex(f) for f in value])
            else:
                return torch.view_as_complex(value)
        else:
            return super().__getattr__(key)


class ComplexDenseTensor(ComplexHandler, DenseTensor, name='ComplexDense'):
    """Complex Dense Factorization
    """
    _complex_params = ['tensor']

    @classmethod
    def new(cls, shape, rank=None, device=None, dtype=torch.cfloat, **kwargs):
        return super().new(shape, rank, device=device, dtype=dtype, **kwargs)


class ComplexTuckerTensor(ComplexHandler, TuckerTensor, name='ComplexTucker'):
    """Complex Tucker Factorization
    """
    _complex_params = ['core', 'factors']

    @classmethod
    def new(cls, shape, rank='same', fixed_rank_modes=None,
            device=None, dtype=torch.cfloat, **kwargs):
        return super().new(shape, rank, fixed_rank_modes=fixed_rank_modes,
            device=device, dtype=dtype, **kwargs)


class ComplexTTTensor(ComplexHandler, TTTensor, name='ComplexTT'):
    """Complex TT Factorization
    """
    _complex_params = ['factors']

    @classmethod
    def new(cls, shape, rank='same', fixed_rank_modes=None,
            device=None, dtype=torch.cfloat, **kwargs):
        return super().new(shape, rank,
            device=device, dtype=dtype, **kwargs)

class ComplexCPTensor(ComplexHandler, CPTensor, name='ComplexCP'):
    """Complex CP Factorization
    """
    _complex_params = ['weights', 'factors']

    @classmethod
    def new(cls, shape, rank='same', fixed_rank_modes=None,
            device=None, dtype=torch.cfloat, **kwargs):
        return super().new(shape, rank,
            device=device, dtype=dtype, **kwargs)
