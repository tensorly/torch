
import torch
from torch import nn

import tensorly as tl
tl.set_backend('pytorch')
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
                super().__setattr__('_' + key, value)
            elif isinstance(value, nn.Parameter):
                self.register_parameter(key, value)
            else:
                self.register_buffer(key, value)
        else:
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

    def register_parameter(self, key, value):
        if key in self._complex_params:
            key = '_' + key
            value = nn.Parameter(torch.view_as_real(value))
        super().register_parameter(key, value)

    def register_buffer(self, key, value):
        if key in self._complex_params:
            key = '_' + key
            value = torch.view_as_real(value)
        super().register_buffer(key, value)


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
