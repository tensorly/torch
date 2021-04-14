import math

import numpy as np
import torch
from torch import nn

import tensorly as tl
tl.set_backend('pytorch')
from tensorly import tenalg
from tensorly.decomposition import parafac, tucker, tensor_train, tensor_train_matrix

from .core import TensorizedMatrix
from ..utils.parameter_list import FactorList

# Author: Jean Kossaifi
# License: BSD 3 clause


def _ensure_tuple(value):
    """Returns a tuple if `value` isn't one already"""
    if isinstance(value, int):
        if value == 1:
            return ()
        else:
            return (value, )
    elif isinstance(value, tuple):
        if value == (1,):
            return ()
        return tuple(value)
    else:
        return tuple(value)


class CPMatrix(TensorizedMatrix, name='CP'):
    """Tensorized Matrix in CP Form

    """
    def __init__(self, weights, factors, tensorized_row_shape, tensorized_column_shape, rank=None, n_matrices=()):
        super().__init__()
        if rank is None:
            _, self.rank = tl.cp_tensor._validate_cp_tensor((weights, factors))
        else:
            self.rank = rank
        self.shape = (np.prod(tensorized_row_shape), np.prod(tensorized_column_shape))
        self.tensorized_shape = tensorized_row_shape + tensorized_column_shape
        self.tensorized_row_shape = tensorized_row_shape
        self.tensorized_column_shape = tensorized_column_shape
        
        self.n_matrices = _ensure_tuple(n_matrices)
        self.order = len(factors)
        self.weights = weights
        self.factors = factors

    @classmethod
    def new(cls, tensorized_row_shape, tensorized_column_shape, rank, n_matrices=(), **kwargs):
        n_matrices = _ensure_tuple(n_matrices)
        tensor_shape = n_matrices + tensorized_row_shape + tensorized_column_shape
        rank = tl.cp_tensor.validate_cp_rank(tensor_shape, rank)

        # Register the parameters
        weights = nn.Parameter(torch.Tensor(rank))
        # Avoid the issues with ParameterList
        factors = [nn.Parameter(torch.Tensor(s, rank)) for s in tensor_shape]

        return cls(weights, factors, tensorized_row_shape, tensorized_column_shape, rank=rank, n_matrices=n_matrices)
    
    @classmethod
    def from_tensor(cls, tensor, tensorized_row_shape, tensorized_column_shape, rank, n_matrices=(), init='random', **kwargs):
        n_matrices = _ensure_tuple(n_matrices)
        rank = tl.cp_tensor.validate_cp_rank(n_matrices + tensorized_row_shape + tensorized_column_shape, rank)

        with torch.no_grad():
            weights, factors = parafac(tensor, rank, **kwargs)
        weights = nn.Parameter(weights)
        factors = [nn.Parameter(f) for f in factors]

        return cls(weights, factors, tensorized_row_shape, tensorized_column_shape, rank, n_matrices)

    @classmethod
    def from_matrix(cls, matrix, tensorized_row_shape, tensorized_column_shape, rank, **kwargs):
        if matrix.ndim > 2:
            n_matrices = _ensure_tuple(tl.shape(matrix)[:-2])
        else:
            n_matrices = ()

        tensor = matrix.reshape((*n_matrices, *tensorized_row_shape, *tensorized_column_shape))
        return cls.from_tensor(tensor, tensorized_row_shape, tensorized_column_shape, rank, n_matrices=n_matrices, **kwargs)

    def init_from_tensor(self, tensor, **kwargs):
        with torch.no_grad():
            weights, factors = parafac(tensor, self.rank, **kwargs)
        self.weights = nn.Parameter(weights)
        self.factors = FactorList([nn.Parameter(f) for f in factors])
        return self

    def init_from_matrix(self, matrix, **kwargs):
        tensor = matrix.reshape((*self.n_matrices, *self.tensorized_row_shape, *self.tensorized_column_shape))
        return self.init_from_tensor(tensor, **kwargs)


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
            return self.__class__(weights, factors, self.tensorized_row_shape, 
                                    self.tensorized_column_shape, n_matrices=self.n_matrices[1:])

        elif isinstance(indices, slice):
            # Index part of a factor
            mixing_factor, *factors = self.factors
            factors = [mixing_factor[indices], *factors]
            weights = self.weights
            return self.__class__(weights, factors, self.tensorized_row_shape, 
                                    self.tensorized_column_shape, n_matrices=self.n_matrices[1:])
            
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

            return self.__class__(weights, index_factors+factors, self.shape, self.tensorized_row_shape, 
                                  self.tensorized_column_shape, n_matrices=self.n_matrices[len(indices):])


class TuckerMatrix(TensorizedMatrix, name='Tucker'):
    """Tensorized Matrix in Tucker Form

    """
    def __init__(self, core, factors, tensorized_row_shape, tensorized_column_shape, rank=None, n_matrices=()):
        super().__init__()
        if rank is None:
            _, self.rank = tl.tucker_tensor._validate_tucker_tensor((core, factors))
        else:
            self.rank = rank
        self.order = self.n_factors = len(factors)
        self.shape = (np.prod(tensorized_row_shape), np.prod(tensorized_column_shape))
        self.tensorized_row_shape = tensorized_row_shape
        self.tensorized_column_shape = tensorized_column_shape

        self.n_matrices = _ensure_tuple(n_matrices)

        setattr(self, 'core', core)
        self.factors = FactorList(factors)
    
    @classmethod
    def new(cls, tensorized_row_shape, tensorized_column_shape, rank, n_matrices=(), **kwargs):
        n_matrices = _ensure_tuple(n_matrices)
        full_shape = n_matrices + tensorized_row_shape + tensorized_column_shape
        rank = tl.tucker_tensor.validate_tucker_rank(full_shape, rank)

        core = nn.Parameter(torch.Tensor(*rank))
        factors = [nn.Parameter(torch.Tensor(s, r)) for (s, r) in zip(full_shape, rank)]
        return cls(core, factors, tensorized_row_shape, tensorized_column_shape, rank, n_matrices=n_matrices)
    
    @classmethod
    def from_tensor(cls, tensor, tensorized_row_shape, tensorized_column_shape, rank, n_matrices=(), **kwargs):
        n_matrices = _ensure_tuple(n_matrices)
        rank = tl.tucker_tensor.validate_tucker_rank(n_matrices + tensorized_row_shape + tensorized_column_shape, rank)

        with torch.no_grad():
            core, factors = tucker(tensor, rank, **kwargs)
        
        return cls(nn.Parameter(core), [nn.Parameter(f) for f in factors], tensorized_row_shape, tensorized_column_shape, rank, n_matrices=n_matrices)
    
    @classmethod
    def from_matrix(cls, matrix, tensorized_row_shape, tensorized_column_shape, rank, **kwargs):
        if matrix.ndim > 2:
            n_matrices = _ensure_tuple(tl.shape(matrix)[:-2])
        else:
            n_matrices = ()

        tensor = matrix.reshape((*n_matrices, *tensorized_row_shape, *tensorized_column_shape))
        return cls.from_tensor(tensor, tensorized_row_shape, tensorized_column_shape, rank, n_matrices=n_matrices, **kwargs)

    def init_from_tensor(self, tensor, init='svd', **kwargs):
        with torch.no_grad():
            core, factors = tucker(tensor, self.rank, **kwargs)
        
        self.core = nn.Parameter(core)
        self.factors = FactorList([nn.Parameter(f) for f in factors])

        return self

    def init_from_matrix(self, matrix, **kwargs):
        tensor = matrix.reshape((*self.n_matrices, *self.tensorized_row_shape, *self.tensorized_column_shape))
        return self.init_from_tensor(tensor, **kwargs)

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
            return self.__class__(core, factors, self.tensorized_row_shape, 
                                  self.tensorized_column_shape, n_matrices=self.n_matrices[1:])

        elif isinstance(indices, slice):
            mixing_factor, *factors = self.factors
            factors = [mixing_factor[indices], *factors]
            return self.__class__(self.core, factors, self.tensorized_row_shape, 
                                  self.tensorized_column_shape, n_matrices=self.n_matrices[1:])
        
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
                return self.__class__(core, factors, self.tensorized_row_shape, 
                                      self.tensorized_column_shape, n_matrices=self.n_matrices[len(indices):])

            # Fully contracted tensor
            return core


class TTTensorized(TensorizedMatrix, name='TT'):
    """Tensorized Matrix in Tensor-Train (MPS) Form

    Notes
    -----
    It may be preferable to use TTMatrix instead

    See Also
    --------
    TTMatrix
    """
    def __init__(self, factors, tensorized_row_shape, tensorized_column_shape, rank=None, n_matrices=()):
        super().__init__()
        if rank is None:
            _, self.rank = tl.tt_tensor._validate_tt_tensor(factors)
        else:
            self.rank = rank
        self.order = self.n_factors = len(factors)
        self.shape = (np.prod(tensorized_row_shape), np.prod(tensorized_column_shape))
        self.tensorized_row_shape = tensorized_row_shape
        self.tensorized_column_shape = tensorized_column_shape

        self.n_matrices = _ensure_tuple(n_matrices)
        self.factors = FactorList(factors)
            
    @classmethod
    def new(cls, tensorized_row_shape, tensorized_column_shape, rank, n_matrices=(), **kwargs):
        n_matrices = _ensure_tuple(n_matrices)
        full_shape = n_matrices + tensorized_row_shape + tensorized_column_shape
        rank = tl.tt_tensor.validate_tt_rank(full_shape, rank)

        # Avoid the issues with ParameterList
        factors = [nn.Parameter(torch.Tensor(rank[i], s, rank[i+1])) for i, s in enumerate(full_shape)]

        return cls(factors, tensorized_row_shape, tensorized_column_shape, rank, n_matrices=n_matrices)
    
    @classmethod
    def from_tensor(cls, tensor, tensorized_row_shape, tensorized_column_shape, rank='same', **kwargs):
        full_shape = tensorized_row_shape + tensorized_column_shape
        n_matrices = _ensure_tuple(tensor.shape[:-len(full_shape)])
        rank = tl.tt_tensor.validate_tt_rank(n_matrices + full_shape, rank)

        with torch.no_grad():
            factors = tensor_train(tensor, rank, **kwargs)
        
        return cls([nn.Parameter(f) for f in factors], tensorized_row_shape, tensorized_column_shape, rank=rank, n_matrices=n_matrices)
    
    @classmethod
    def from_matrix(cls, matrix, tensorized_row_shape, tensorized_column_shape, rank,  **kwargs):
        if matrix.ndim > 2:
            n_matrices = _ensure_tuple(tl.shape(matrix)[:-2])
        else:
            n_matrices=(),
        tensor = matrix.reshape((*n_matrices, *tensorized_row_shape, *tensorized_column_shape))
        return cls.from_tensor(tensor, tensorized_row_shape, tensorized_column_shape, rank, **kwargs)

    def init_from_tensor(self, tensor, **kwargs):
        with torch.no_grad():
            factors = tensor_train(tensor, self.rank, **kwargs)
        
        self.factors = FactorList([nn.Parameter(f) for f in factors])
        return self

    def init_from_matrix(self, matrix, **kwargs):
        tensor = matrix.reshape((*self.n_matrices, *self.tensorized_row_shape, *self.tensorized_column_shape))
        return self.init_from_tensor(tensor, **kwargs)

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
            return self.__class__([next_factor, *factors], self.tensorized_row_shape, 
                                      self.tensorized_column_shape, n_matrices=self.n_matrices[1:])
        
        elif isinstance(indices, slice):
            mixing_factor, *factors = self.factors
            factors = [mixing_factor[:, indices], *factors]
            return self.__class__(factors, self.tensorized_row_shape, 
                                      self.tensorized_column_shape, n_matrices=self.n_matrices[1:])

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

            # We have contracted all cores, so have a 2D matrix
            if factor.ndim == 2:
                if self.order == (i+1):
                    # No factors left
                    return factor.squeeze()
                else:
                    next_factor, *factors = self.factors[i+1:]
                    factor = tenalg.mode_dot(next_factor, factor, 0)
                    return self.__class__([factor, *factors], self.tensorized_row_shape, 
                                      self.tensorized_column_shape,
                                      n_matrices=self.n_matrices[len(indices):])
            else:
                return self.__class__([*factors, factor, *self.factors[i+1:]], self.tensorized_row_shape, 
                                      self.tensorized_column_shape,
                                      n_matrices=self.n_matrices[len(indices):])


class TTMatrix(TensorizedMatrix, name='TTM'):
    """Tensorized Matrix in the Tensor-Train Matrix (MPO) Form
    """
    def __init__(self, factors, tensorized_row_shape, tensorized_column_shape, rank=None, n_matrices=1):
        super().__init__()

        if rank is None:
            _, self.rank = tl.tt_matrix._validate_tt_matrix(factors)
        
        self.tensorized_row_shape = tensorized_row_shape
        self.tensorized_column_shape = tensorized_column_shape
        self.tensorized_shape = tensorized_row_shape + tensorized_column_shape
        self.shape = (np.prod(tensorized_row_shape), np.prod(tensorized_column_shape))
        self.order = len(tensorized_row_shape)

        self.factors = FactorList(factors)
        self.rank = rank
        self.n_matrices = _ensure_tuple(n_matrices)

    @classmethod
    def new(cls, tensorized_row_shape, tensorized_column_shape, rank, n_matrices=(), **kwargs):
        n_matrices = _ensure_tuple(n_matrices)
        shape = tensorized_row_shape + tensorized_column_shape
        rank = tl.tt_matrix.validate_tt_matrix_rank(shape, rank)

        if not n_matrices:
            factors = [nn.Parameter(torch.Tensor(rank[i], tensorized_row_shape[i], tensorized_column_shape[i], rank[i + 1]))\
                        for i in range(len(tensorized_row_shape))]
        elif len(n_matrices) == 1:
            factors = [nn.Parameter(torch.Tensor(n_matrices[0], rank[i], tensorized_row_shape[i], tensorized_column_shape[i], rank[i + 1]))\
                        for i in range(len(tensorized_row_shape))]
        else:
            raise ValueError(f'Currently a single dimension is supported for n_matrices, it should an integer (by default, 1) but got n_matrices={n_matrices}.')
        
        return cls(factors, tensorized_row_shape, tensorized_column_shape, rank=rank, n_matrices=n_matrices)
    
    @classmethod
    def from_tensor(cls, tensor, tensorized_row_shape, tensorized_column_shape, rank, n_matrices=(), **kwargs):
        rank = tl.tt_matrix.validate_tt_matrix_rank(tensorized_row_shape + tensorized_column_shape, rank)

        if n_matrices == ():
            with torch.no_grad():
                factors = tensor_train_matrix(tensor, rank, **kwargs)
            factors = [nn.Parameter(f) for f in factors]

        else:
            factors = [torch.zeros(n_matrices[0], rank[i], tensorized_row_shape[i], tensorized_column_shape[i], rank[i + 1])\
                        for i in range(len(tensorized_row_shape))]
            for i in range(n_matrices[0]):
                with torch.no_grad():
                    factors_i = tensor_train_matrix(tensor[i], rank, **kwargs)
                    print(factors_i)
                    for j, factor in enumerate(factors_i):
                        factors[j][i, ...] = factor
            factors = [nn.Parameter(f) for f in factors]
        return cls(factors, tensorized_row_shape, tensorized_column_shape, rank, n_matrices)

    @classmethod
    def from_matrix(cls, matrix, tensorized_row_shape, tensorized_column_shape, rank, **kwargs):
        if matrix.ndim > 2:
            n_matrices = _ensure_tuple(tl.shape(matrix)[:-2])
        else:
            n_matrices = ()
        tensor = matrix.reshape((*n_matrices, *tensorized_row_shape, *tensorized_column_shape))
        return cls.from_tensor(tensor, tensorized_row_shape, tensorized_column_shape, rank, n_matrices=n_matrices, **kwargs)

    def init_from_tensor(self, tensor, **kwargs):
        if self.n_matrices == ():
            with torch.no_grad():
                factors = tensor_train_matrix(tensor, self.rank, **kwargs)
            factors = [nn.Parameter(f) for f in factors]

        else:
            factors = [torch.zeros(self.n_matrices[0], self.rank[i], self.tensorized_row_shape[i], self.tensorized_column_shape[i], self.rank[i + 1])\
                        for i in range(len(self.tensorized_row_shape))]
            for i in range(self.n_matrices[0]):
                with torch.no_grad():
                    factors_i = tensor_train_matrix(tensor[i], self.rank, **kwargs)
                    print(factors_i)
                    for j, factor in enumerate(factors_i):
                        factors[j][i, ...] = factor
            factors = [nn.Parameter(f) for f in factors]
        
        self.factors = FactorList(factors)
        return self

    def init_from_matrix(self, matrix, **kwargs):
        tensor = matrix.reshape((*self.n_matrices, *self.tensorized_row_shape, *self.tensorized_column_shape))
        return self.init_from_tensor(tensor, **kwargs)

    @property
    def decomposition(self):
        return self.factors

    def to_tensor(self):
        if not self.n_matrices:
            return tl.tt_matrix_to_tensor(self.decomposition)
        else:
            ten = tl.tt_matrix_to_tensor(self[0].decomposition)
            res = torch.zeros(*self.n_matrices, *ten.shape)
            res[0, ...] = ten
            for i in range(1, self.n_matrices[0]):
                res[i, ...] = tl.tt_matrix_to_tensor(self[i].decomposition)
            return res

    def normal_(self, mean=0, std=1):
        if mean != 0:
            raise ValueError(f'Currently only mean=0 is supported, but got mean={mean}')

        r = np.prod(self.rank)
        std_factors = (std/r)**(1/self.order)
        with torch.no_grad():
            for factor in self.factors:
                factor.data.normal_(0, std_factors)
        return self

    def to_matrix(self):
        if not self.n_matrices:
            return tl.tt_matrix_to_matrix(self.decomposition)
        else:
            res = torch.zeros(*(self.n_matrices + self.shape))
            for i in range(self.n_matrices[0]):
                res[i, ...] = tl.tt_matrix_to_matrix(self[i].decomposition)
            return res

    def __getitem__(self, indices):
        if not isinstance(indices, int) or not self.n_matrices:
            raise ValueError(f'Currently only indexing over n_matrices is supported for TTMatrices.')

        return self.__class__([f[indices, ...] for f in self.factors],
                              self.tensorized_row_shape, self.tensorized_column_shape, self.rank, self.n_matrices[1:])

    def __torch_function__(self, func, types, args=(), kwargs=None):
        if kwargs is None:
            kwargs = {}

        args = [t.to_matrix() if hasattr(t, 'to_matrix') else t for t in args]
        return func(*args, **kwargs)
