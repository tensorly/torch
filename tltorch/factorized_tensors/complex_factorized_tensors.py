
import torch
from torch import nn

import tensorly as tl
tl.set_backend('pytorch')
from tltorch.factorized_tensors.factorized_tensors import TuckerTensor, CPTensor, TTTensor, DenseTensor
from tltorch.utils.parameter_list import FactorList, ComplexFactorList


# Author: Jean Kossaifi
# License: BSD 3 clause

class ComplexHandler():
    def __setattr__(self, key, value):
        if isinstance(value, (FactorList)):
            value = ComplexFactorList(value)
            super().__setattr__(key, value)
            
        elif isinstance(value, nn.Parameter):
            self.register_parameter(key, value)
        elif torch.is_tensor(value):
            self.register_buffer(key, value)
        else:
            super().__setattr__(key, value)

    def __getattr__(self, key):
        value = super().__getattr__(key)
        if torch.is_tensor(value):
            value = torch.view_as_complex(value)
        return value

    def register_parameter(self, key, value):
        value = nn.Parameter(torch.view_as_real(value))
        super().register_parameter(key, value)

    def register_buffer(self, key, value):
        value = torch.view_as_real(value)
        super().register_buffer(key, value)


class ComplexDenseTensor(ComplexHandler, DenseTensor, name='ComplexDense'):
    """Complex Dense Factorization
    """
    @classmethod
    def new(cls, shape, rank=None, device=None, dtype=torch.cfloat, **kwargs):
        return super().new(shape, rank, device=device, dtype=dtype, **kwargs)

class ComplexTuckerTensor(ComplexHandler, TuckerTensor, name='ComplexTucker'):
    """Complex Tucker Factorization
    """
    @classmethod
    def new(cls, shape, rank='same', fixed_rank_modes=None,
            device=None, dtype=torch.cfloat, **kwargs):
        return super().new(shape, rank, fixed_rank_modes=fixed_rank_modes,
            device=device, dtype=dtype, **kwargs)

class ComplexTTTensor(ComplexHandler, TTTensor, name='ComplexTT'):
    """Complex TT Factorization
    """
    @classmethod
    def new(cls, shape, rank='same', fixed_rank_modes=None,
            device=None, dtype=torch.cfloat, **kwargs):
        return super().new(shape, rank,
            device=device, dtype=dtype, **kwargs)

class ComplexCPTensor(ComplexHandler, CPTensor, name='ComplexCP'):
    """Complex CP Factorization
    """
    @classmethod
    def new(cls, shape, rank='same', fixed_rank_modes=None,
            device=None, dtype=torch.cfloat, **kwargs):
        return super().new(shape, rank,
            device=device, dtype=dtype, **kwargs)
