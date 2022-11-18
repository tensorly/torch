
import math

import numpy as np
import torch
from torch import nn

import tensorly as tl
tl.set_backend('pytorch')
from tensorly import tenalg
from tensorly.decomposition import tucker, tensor_train

from tltorch.factorized_tensors.core import FactorizedTensor
from tltorch.utils import FactorList


# Author: Jean Kossaifi
# License: BSD 3 clause


class ComplexTuckerTensor(FactorizedTensor, name='ComplexTucker'):
    """Tucker Factorization

    Parameters
    ----------
    core
    factors
    shape
    rank
    """
    def __init__(self, core, factors, shape=None, rank=None):
        super().__init__()
        if shape is not None and rank is not None:
            self.shape, self.rank = shape, rank
        else:
            core = torch.view_as_complex(core)
            factors = [torch.view_as_complex(f) for f in factors]
            self.shape, self.rank = tl.tucker_tensor._validate_tucker_tensor((core, factors))
            core = torch.view_as_real(core)
            factors = [torch.view_as_real(f) for f in factors]

        self.order = len(self.shape)
        setattr(self, 'core', core)
        self.factors = FactorList(factors)
    
    @classmethod
    def new(cls, shape, rank, fixed_rank_modes=None,
            device=None, dtype=None, **kwargs):
        rank = tl.tucker_tensor.validate_tucker_rank(shape, rank, fixed_modes=fixed_rank_modes)

        # Register the parameters
        core = nn.Parameter(torch.empty((*rank, 2), device=device, dtype=dtype))
        # Avoid the issues with ParameterList
        factors = [nn.Parameter(torch.empty((s, r, 2), device=device, dtype=dtype)) for (s, r) in zip(shape, rank)]

        return cls(core, factors)

    @classmethod
    def from_tensor(cls, tensor, rank='same', fixed_rank_modes=None, **kwargs):
        shape = tensor.shape
        rank = tl.tucker_tensor.validate_tucker_rank(shape, rank, fixed_modes=fixed_rank_modes)

        with torch.no_grad():
            core, factors = tucker(tensor, rank, **kwargs)
        
        return cls(nn.Parameter(core.contiguous()), [nn.Parameter(f.contiguous()) for f in factors])

    def init_from_tensor(self, tensor, unsqueezed_modes=None, unsqueezed_init='average', **kwargs):
        """Initialize the tensor factorization from a tensor

        Parameters
        ----------
        tensor : torch.Tensor
            full tensor to decompose
        unsqueezed_modes : int list
            list of modes for which the rank is 1 that don't correspond to a mode in the full tensor
            essentially we are adding a new dimension for which the core has dim 1, 
            and that is not initialized through decomposition.
            Instead first `tensor` is decomposed into the other factors. 
            The `unsqueezed factors` are then added and  initialized e.g. with 1/dim[i]
        unsqueezed_init : 'average' or float
            if unsqueezed_modes, this is how the added "unsqueezed" factors will be initialized
            if 'average', then unsqueezed_factor[i] will have value 1/tensor.shape[i]
        """
        if unsqueezed_modes is not None:
            unsqueezed_modes = sorted(unsqueezed_modes)
            for mode in unsqueezed_modes[::-1]:
                if self.rank[mode] != 1:
                    msg = 'It is only possible to initialize by averagig over mode for which rank=1.'
                    msg += f'However, got unsqueezed_modes={unsqueezed_modes} but rank[{mode}]={self.rank[mode]} != 1.'
                    raise ValueError(msg)
                        
            rank = tuple(r for (i, r) in enumerate(self.rank) if i not in unsqueezed_modes)
        else:
            rank = self.rank

        with torch.no_grad():
            core, factors = tucker(tensor, rank, **kwargs)
            
            if unsqueezed_modes is not None:
                # Initialise with 1/shape[mode] or given value
                for mode in unsqueezed_modes:
                    size = self.shape[mode]
                    factor = torch.ones(size, 1)
                    if unsqueezed_init == 'average':
                        factor /= size
                    else:
                        factor *= unsqueezed_init
                    factors.insert(mode, factor)
                    core = core.unsqueeze(mode)

        self.core = nn.Parameter(torch.view_as_real(core).contiguous())
        self.factors = FactorList([nn.Parameter(torch.view_as_real(f).contiguous()) for f in factors])
        return self

    @property
    def decomposition(self):
        return self.core, self.factors

    @property
    def decomposition_as_complex(self):
        return torch.view_as_complex(self.core), [torch.view_as_complex(f) for f in self.factors]

    def to_tensor(self):
        return torch.view_as_real(tl.tucker_to_tensor(self.decomposition_as_complex))

    def normal_(self, mean=0, std=1):
        if mean != 0:
            raise ValueError(f'Currently only mean=0 is supported, but got mean={mean}')
            
        r = np.prod([math.sqrt(r) for r in self.rank])
        std_factors = (std/r)**(1/(self.order+1))
        
        with torch.no_grad():
            self.core.data.normal_(0, std_factors)
            for factor in self.factors:
                factor.data.normal_(0, std_factors)
        return self

    def __getitem__(self, indices):
        core, factors = self.decomposition_as_complex
        if isinstance(indices, int):
            # Select one dimension of one mode
            mixing_factor, *factors = factors
            core = tenalg.mode_dot(torch.view_as_complex(core), torch.view_as_complex(mixing_factor)[indices, :], 0)
            return torch.view_as_real(core), self.factors[1:]
        
        elif isinstance(indices, slice):
            mixing_factor, *factors = self.factors
            factors = [mixing_factor[indices], *factors]
            return self.__class__(self.core, factors)
        
        else:
            # Index multiple dimensions
            modes = []
            factors = []
            factors_contract = []
            for i, (index, factor) in enumerate(zip(indices, self.factors)):
                if index is Ellipsis:
                    raise ValueError(f'Ellipsis is not yet supported, yet got indices={indices}, indices[{i}]={index}.')
                if isinstance(index, int):
                    modes.append(i)
                    factors_contract.append(torch.view_as_complex(factor)[index, :])
                else:
                    factors.append(torch.view_as_complex(factor)[index, :])

            core = tenalg.multi_mode_dot(torch.view_as_complex(self.core), factors_contract, modes=modes)
            core = torch.view_as_real(core)
            factors = [torch.view_as_real(f) for f in factors] + self.factors[i+1:]

            if factors:
                return self.__class__(core, factors)

            # Fully contracted tensor
            return core


class ComplexTTTensor(FactorizedTensor, name='ComplexTT'):
    """Tensor-Train (Matrix-Product-State) Factorization
    """
    def __init__(self, factors, shape=None, rank=None):
        super().__init__()
        if shape is None or rank is None:
            self.shape, self.rank = tl.tt_tensor._validate_tt_tensor([torch.view_as_complex(f) for f in factors])
        else:
            self.shape, self.rank = shape, rank
        
        self.order = len(self.shape)
        self.factors = FactorList(factors)
    
    @classmethod
    def new(cls, shape, rank, device=None, dtype=None, **kwargs):
        rank = tl.tt_tensor.validate_tt_rank(shape, rank)

        # Avoid the issues with ParameterList
        factors = [nn.Parameter(torch.empty((rank[i], s, rank[i+1], 2), device=device, dtype=dtype)) for i, s in enumerate(shape)]

        return cls(factors)

    @classmethod
    def from_tensor(cls, tensor, rank='same', **kwargs):
        shape = tensor.shape
        rank = tl.tt_tensor.validate_tt_rank(shape, rank)

        with torch.no_grad():
            # TODO: deal properly with wrong kwargs
            factors = tensor_train(tensor, rank)
        
        return cls([nn.Parameter(f.contiguous()) for f in factors])

    def init_from_tensor(self, tensor, **kwargs):
        with torch.no_grad():
            # TODO: deal properly with wrong kwargs
            factors = tensor_train(tensor, self.rank)
        
        self.factors = FactorList([nn.Parameter(f.contiguous()) for f in factors])
        self.rank = tuple([f.shape[0] for f in factors] + [1])
        return self

    @property
    def decomposition(self):
        return self.factors

    @property
    def decomposition_as_complex(self):
        return [torch.view_as_complex(f) for f in self.factors]

    def to_tensor(self):
        return torch.view_as_real(tl.tt_to_tensor(self.decomposition_as_complex))

    def normal_(self,  mean=0, std=1):
        if mean != 0:
            raise ValueError(f'Currently only mean=0 is supported, but got mean={mean}')

        r = np.prod(self.rank)
        std_factors = (std/r)**(1/self.order)
        with torch.no_grad():
            for factor in self.factors:
                factor.data.normal_(0, std_factors)
        return self

    def __getitem__(self, indices):
        if isinstance(indices, int):
            # Select one dimension of one mode
            factor, next_factor, *factors = self.factors
            next_factor = tenalg.mode_dot(torch.view_as_complex(next_factor), torch.view_as_complex(factor[:, indices, :,:].squeeze(1)), 0)
            return ComplexTTTensor([torch.view_as_real(next_factor), *factors])

        elif isinstance(indices, slice):
            mixing_factor, *factors = self.factors
            factors = [mixing_factor[:, indices, :, :], *factors]
            return ComplexTTTensor(factors)

        else:
            factors = []
            complex_factors = self.decomposition_as_complex
            all_contracted = True
            for i, index in enumerate(indices):
                if index is Ellipsis:
                    raise ValueError(f'Ellipsis is not yet supported, yet got indices={indices}, indices[{i}]={index}.')
                if isinstance(index, int):
                    if i:
                        factor = tenalg.mode_dot(factor, complex_factors[i][:, index, :].T, -1)
                    else:
                        factor = complex_factors[i][:, index, :]
                else:
                    if i:
                        if all_contracted:
                            factor = tenalg.mode_dot(complex_factors[i][:, index, :], factor, 0)
                        else:
                            factors.append(factor)
                            factor = complex_factors[i][:, index, :]
                    else:
                        factor = complex_factors[i][:, index, :]
                    all_contracted = False

            if factor.ndim == 2: # We have contracted all cores, so have a 2D matrix
                if self.order == (i+1):
                    # No factors left
                    return factor.squeeze()
                else:
                    next_factor, *factors = complex_factors[i+1:]
                    factor = tenalg.mode_dot(next_factor, factor, 0)
                    return ComplexTTTensor([torch.view_as_real(factor)] + [torch.view_as_real(f) for f in factors])
            else:
                return ComplexTTTensor([torch.view_as_real(f) for f in factors] + [torch.view_as_real(factor)] + self.factors[i+1:])
