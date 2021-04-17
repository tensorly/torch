import math

import numpy as np
import torch
from torch import nn

import tensorly as tl
tl.set_backend('pytorch')
from tensorly import tenalg
from tensorly.decomposition import parafac, tucker, tensor_train, tensor_train_matrix

from .core import FactorizedTensor
from ..utils import FactorList


# Author: Jean Kossaifi
# License: BSD 3 clause


class CPTensor(FactorizedTensor, name='CP'):
    """CP Factorization

    Parameters
    ----------
    weights
    factors
    shape
    rank
    """

    def __init__(self, weights, factors, shape=None, rank=None):
        super().__init__()
        if shape is not None and rank is not None:
            self.shape, self.rank = shape, rank
        else:
            self.shape, self.rank = tl.cp_tensor._validate_cp_tensor((weights, factors))
        self.order = len(self.shape)

        self.weights = weights
        self.factors = FactorList(factors)
    
    @classmethod
    def new(cls, shape, rank, **kwargs):
        rank = tl.cp_tensor.validate_cp_rank(shape, rank)

        # Register the parameters
        weights = nn.Parameter(torch.Tensor(rank))
        # Avoid the issues with ParameterList
        factors = [nn.Parameter(torch.Tensor(s, rank)) for s in shape]

        return cls(weights, factors)

    @classmethod
    def from_tensor(cls, tensor, rank='same', **kwargs):
        shape = tensor.shape
        rank = tl.cp_tensor.validate_cp_rank(shape, rank)
        dtype = tensor.dtype

        with torch.no_grad():
            weights, factors = parafac(tensor.to(torch.float64), rank, **kwargs)
        
        return cls(nn.Parameter(weights.to(dtype)), [nn.Parameter(f.to(dtype)) for f in factors])

    def init_from_tensor(self, tensor, l2_reg=1e-5, **kwargs):
        with torch.no_grad():
            weights, factors = parafac(tensor, self.rank, l2_reg=l2_reg, **kwargs)
        
        self.weights = nn.Parameter(weights)
        self.factors = FactorList([nn.Parameter(f) for f in factors])
        return self

    @property
    def decomposition(self):
        return self.weights, self.factors

    def to_tensor(self):
        return tl.cp_to_tensor(self.decomposition)

    def normal_(self, mean=0, std=1):
        super().normal_(mean, std)
        std_factors = (std/math.sqrt(self.rank))**(1/self.order)

        with torch.no_grad():
            self.weights.fill_(1)
            for factor in self.factors:
                factor.data.normal_(0, std_factors)
        return self
    
    def __getitem__(self, indices):
        if isinstance(indices, int):
            # Select one dimension of one mode
            mixing_factor, *factors = self.factors
            weights = self.weights*mixing_factor[indices, :]
            return self.__class__(weights, factors)

        elif isinstance(indices, slice):
            # Index part of a factor
            mixing_factor, *factors = self.factors
            factors = [mixing_factor[indices], *factors]
            weights = self.weights
            return self.__class__(weights, factors)

        else:
            # Index multiple dimensions
            factors = self.factors
            index_factors = []
            weights = self.weights
            for index in indices:
                if index is Ellipsis:
                    raise ValueError(f'Ellipsis is not yet supported, yet got indices={indices} which contains one.')

                mixing_factor, *factors = factors
                if isinstance(index, int):
                    if factors or index_factors:
                        weights = weights*mixing_factor[index, :]
                    else:
                        # No factors left
                        return tl.sum(weights*mixing_factor[index, :])
                else:
                    index_factors.append(mixing_factor[index])
            
            return self.__class__(weights, index_factors + factors)
        # return self.__class__(*tl.cp_indexing(self.weights, self.factors, indices))

    def transduct(self, new_dim, mode=0, new_factor=None):
        """Transduction adds a new dimension to the existing factorization

        Parameters
        ----------
        new_dim : int
            dimension of the new mode to add
        mode : where to insert the new dimension, after the channels, default is 0
            by default, insert the new dimensions before the existing ones
            (e.g. add time before height and width)

        Returns
        -------
        self
        """
        factors = self.factors
        # Important: don't increment the order before accessing factors which uses order!
        self.order += 1
        self.shape = self.shape[:mode] + (new_dim,) + self.shape[mode:]

        if new_factor is None:
            new_factor = torch.ones(new_dim, self.rank)#/new_dim

        factors.insert(mode, nn.Parameter(new_factor.to(factors[0].device)))
        self.factors = FactorList(factors)

        return self

        
class TuckerTensor(FactorizedTensor, name='Tucker'):
    """Tucker Factorization

    Parameters
    ----------
    core
    factors
    shape
    rank
    """
    def __init__(self, core, factors):
        super().__init__()
        self.shape, self.rank = tl.tucker_tensor._validate_tucker_tensor((core, factors))
        self.order = len(self.shape)

        setattr(self, 'core', core)
        self.factors = FactorList(factors)
    
    @classmethod
    def new(cls, shape, rank, fixed_rank_modes=None, **kwargs):
        rank = tl.tucker_tensor.validate_tucker_rank(shape, rank, fixed_modes=fixed_rank_modes)

        # Register the parameters
        core = nn.Parameter(torch.Tensor(*rank))
        # Avoid the issues with ParameterList
        factors = [nn.Parameter(torch.Tensor(s, r)) for (s, r) in zip(shape, rank)]
        return cls(core, factors)

    @classmethod
    def from_tensor(cls, tensor, rank='same', fixed_rank_modes=None, **kwargs):
        shape = tensor.shape
        rank = tl.tucker_tensor.validate_tucker_rank(shape, rank, fixed_modes=fixed_rank_modes)

        with torch.no_grad():
            core, factors = tucker(tensor, rank, **kwargs)
        
        return cls(nn.Parameter(core.contiguous), [nn.Parameter(f) for f in factors])

    def init_from_tensor(self, tensor, **kwargs):
        with torch.no_grad():
            core, factors = tucker(tensor, self.rank, **kwargs)
        
        self.core = nn.Parameter(core.contiguous())
        self.factors = FactorList([nn.Parameter(f) for f in factors])
        return self

    @property
    def decomposition(self):
        return self.core, self.factors

    def to_tensor(self):
        return tl.tucker_to_tensor(self.decomposition)

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
        if isinstance(indices, int):
            # Select one dimension of one mode
            mixing_factor, *factors = self.factors
            core = tenalg.mode_dot(self.core, mixing_factor[indices, :], 0)
            return TuckerTensor(core, factors)
        
        elif isinstance(indices, slice):
            mixing_factor, *factors = self.factors
            factors = [mixing_factor[indices], *factors]
            return TuckerTensor(self.core, factors)
        
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
                    factors_contract.append(factor[index])
                else:
                    factors.append(factor[index])

            core = tenalg.multi_mode_dot(self.core, factors_contract, modes=modes)
            factors = factors + self.factors[i+1:]

            if factors:
                return TuckerTensor(core, factors)

            # Fully contracted tensor
            return core


class TTTensor(FactorizedTensor, name='TT'):
    """Tensor-Train (Matrix-Product-State) Factorization


    Parameters
    ----------
    factors
    shape
    rank
    """
    def __init__(self, factors, shape=None, rank=None):
        super().__init__()
        if shape is None or rank is None:
            self.shape, self.rank = tl.tt_tensor._validate_tt_tensor(factors)
        else:
            self.shape, self.rank = shape, rank
        
        self.order = len(self.shape)
        self.factors = FactorList(factors)
    
    @classmethod
    def new(cls, shape, rank, **kwargs):
        rank = tl.tt_tensor.validate_tt_rank(shape, rank)

        # Avoid the issues with ParameterList
        factors = [nn.Parameter(torch.Tensor(rank[i], s, rank[i+1])) for i, s in enumerate(shape)]

        return cls(factors)

    @classmethod
    def from_tensor(cls, tensor, rank='same', **kwargs):
        shape = tensor.shape
        rank = tl.tt_tensor.validate_tt_rank(shape, rank)

        with torch.no_grad():
            # TODO: deal properly with wrong kwargs
            factors = tensor_train(tensor, rank)
        
        return cls([nn.Parameter(f) for f in factors])

    def init_from_tensor(self, tensor, **kwargs):
        with torch.no_grad():
            # TODO: deal properly with wrong kwargs
            factors = tensor_train(tensor, self.rank)
        
        self.factors = FactorList([nn.Parameter(f) for f in factors])
        return self

    @property
    def decomposition(self):
        return self.factors

    def to_tensor(self):
        return tl.tt_to_tensor(self.decomposition)

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
            next_factor = tenalg.mode_dot(next_factor, factor[:, indices, :].squeeze(1), 0)
            return TTTensor([next_factor, *factors])

        elif isinstance(indices, slice):
            mixing_factor, *factors = self.factors
            factors = [mixing_factor[:, indices], *factors]
            return TTTensor(factors)

        else:
            factors = []
            all_contracted = True
            for i, index in enumerate(indices):
                if index is Ellipsis:
                    raise ValueError(f'Ellipsis is not yet supported, yet got indices={indices}, indices[{i}]={index}.')
                if isinstance(index, int):
                    if i:
                        factor = tenalg.mode_dot(factor, self.factors[i][:, index, :].T, -1)
                    else:
                        factor = self.factors[i][:, index, :]
                else:
                    if i:
                        if all_contracted:
                            factor = tenalg.mode_dot(self.factors[i][:, index, :], factor, 0)
                        else:
                            factors.append(factor)
                            factor = self.factors[i][:, index, :]
                    else:
                        factor = self.factors[i][:, index, :]
                    all_contracted = False

            if factor.ndim == 2: # We have contracted all cores, so have a 2D matrix
                if self.order == (i+1):
                    # No factors left
                    return factor.squeeze()
                else:
                    next_factor, *factors = self.factors[i+1:]
                    factor = tenalg.mode_dot(next_factor, factor, 0)
                    return TTTensor([factor, *factors])
            else:
                return TTTensor([*factors, factor, *self.factors[i+1:]])

    def transduct(self, new_dim, mode=0, new_factor=None):
        """Transduction adds a new dimension to the existing factorization

        Parameters
        ----------
        new_dim : int
            dimension of the new mode to add
        mode : where to insert the new dimension, after the channels, default is 0
            by default, insert the new dimensions before the existing ones
            (e.g. add time before height and width)

        Returns
        -------
        self
        """
        factors = self.factors

        # Important: don't increment the order before accessing factors which uses order!
        self.order += 1
        new_rank = self.rank[mode]
        self.rank = self.rank[:mode] + (new_rank, ) + self.rank[mode:]
        self.shape = self.shape[:mode] + (new_dim, ) + self.shape[mode:]

        # Init so the reconstruction is equivalent to concatenating the previous self new_dim times
        if new_factor is None:
            new_factor = torch.zeros(new_rank, new_dim, new_rank)
            for i in range(new_dim):
                new_factor[:, i, :] = torch.eye(new_rank)#/new_dim
            # Below: <=> static prediciton
            # new_factor[:, new_dim//2, :] = torch.eye(new_rank)


        factors.insert(mode, nn.Parameter(new_factor.to(factors[0].device)))
        self.factors = FactorList(factors)

        return self


class TTMatrix(FactorizedTensor, name='TTM'):
    """Tensor-Train Matrix (MPO) Factorization

    Parameters
    ----------
    factors
    shape
    rank
    """
    def __init__(self, factors, shape=None, rank=None):
        if shape is None or rank is None:
            self.shape, self.rank = tl.tt_matrix._validate_tt_matrix(factors)
        else:
            self.shape, self.rank = shape, rank

        n_dim = len(self.tensorized_shape) // 2
        tensorized_rows_shape = np.prod(self.tensorized_shape[:n_dim])
        tensorized_columns_shape = np.prod(self.tensorized_shape[n_dim:])
        self.shape = (tensorized_rows_shape, tensorized_columns_shape)
        self.order = len(self.shape)

        self.factors = FactorList(factors)
    
    @classmethod
    def new(cls, tensorized_shape, rank, **kwargs):

        n_dim = len(tensorized_shape) // 2
        if n_dim*2 == len(tensorized_shape):
            n_matrices = 1
        else:
            n_matrices = tensorized_shape[0]
            tensorized_shape = tensorized_shape[1:]

        rank = tl.tt_matrix.validate_tt_matrix_rank(tensorized_shape, rank)

        tensorized_row_shape = tensorized_shape[:n_dim]
        tensorized_column_shape = tensorized_shape[n_dim:]

        if n_matrices == 1:
            factors = [nn.Parameter(torch.Tensor(rank[i], tensorized_row_shape[i], tensorized_column_shape[i], rank[i + 1]))\
                        for i in range(len(tensorized_row_shape))]
        else:
            factors = [nn.Parameter(torch.Tensor(n_matrices, rank[i], tensorized_row_shape[i], tensorized_column_shape[i], rank[i + 1]))\
                        for i in range(len(tensorized_row_shape))]
        
        return cls(factors)

    @classmethod
    def from_tensor(cls, tensor, rank='same', **kwargs):
        tensorized_shape = tensor.shape
        rank = tl.tt_matrix.validate_tt_matrix_rank(tensorized_shape, rank)

        with torch.no_grad():
            factors = tensor_train_matrix(tensor, rank, **kwargs)
        
        return cls([nn.Parameter(f) for f in factors])

    def init_from_tensor(self, tensor, **kwargs):
        with torch.no_grad():
            factors = tensor_train_matrix(tensor, self.rank, **kwargs)
        
        self.factors = FactorList([nn.Parameter(f) for f in factors])
        return self

    @property
    def decomposition(self):
        return self.factors

    def to_tensor(self):
        return tl.tt_matrix_to_matrix(self.decomposition)

    def extra_repr(self):
        return f'shape={self.shape}, tensorized_shape={self.tensorized_shape}, rank={self.rank}, order={self.order}'
    
    def __torch_function__(self, func, types, args=(), kwargs=None):
        if kwargs is None:
            kwargs = {}

        args = [t.to_tensor() if hasattr(t, 'to_tensor') else t for t in args]
        return func(*args, **kwargs)
