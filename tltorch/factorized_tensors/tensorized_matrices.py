import math
from collections import Iterable
import warnings

import numpy as np
import torch
from torch import nn

import tensorly as tl
tl.set_backend('pytorch')
from tensorly import tenalg
from tensorly.decomposition import tensor_train_matrix, parafac, tucker

from .core import TensorizedTensor, _ensure_tuple
from .factorized_tensors import CPTensor, TuckerTensor
from ..utils.parameter_list import FactorList

einsum_symbols = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
einsum_symbols_set = set(einsum_symbols)


# Author: Jean Kossaifi
# License: BSD 3 clause

def is_tensorized_shape(shape):
    """Checks if a given shape represents a tensorized tensor."""
    if all(isinstance(s, int) for s in shape):
        return False
    return True

def tensorized_shape_to_shape(tensorized_shape):
    return [s if isinstance(s, int) else np.prod(s) for s in tensorized_shape]

class CPTensorized(CPTensor, TensorizedTensor, name='CP'):
    
    def __init__(self, weights, factors, tensorized_shape, rank=None):
        tensor_shape = sum([(e,) if isinstance(e, int) else tuple(e) for e in tensorized_shape], ())

        super().__init__(weights, factors, tensor_shape, rank)

        # Modify only what varies from the Tensor case
        self.shape = tensorized_shape_to_shape(tensorized_shape)
        # self.tensor_shape = tensor_shape
        self.order = len(tensor_shape)
        self.tensorized_shape = tensorized_shape

    @classmethod
    def new(cls, tensorized_shape, rank, device=None, dtype=None, **kwargs):
        flattened_tensorized_shape = sum([[e] if isinstance(e, int) else list(e) for e in tensorized_shape], [])
        rank = tl.cp_tensor.validate_cp_rank(flattened_tensorized_shape, rank)

        # Register the parameters
        weights = nn.Parameter(torch.empty(rank, device=device, dtype=dtype))
        # Avoid the issues with ParameterList
        factors = [nn.Parameter(torch.empty((s, rank), device=device, dtype=dtype)) for s in flattened_tensorized_shape]

        return cls(weights, factors, tensorized_shape, rank=rank)

    @classmethod
    def from_tensor(cls, tensor, tensorized_shape, rank='same', **kwargs):
        shape = tensor.shape
        rank = tl.cp_tensor.validate_cp_rank(shape, rank)
        dtype = tensor.dtype

        with torch.no_grad():
            weights, factors = parafac(tensor.to(torch.float64), rank, **kwargs)
        
        return cls(nn.Parameter(weights.to(dtype)), [nn.Parameter(f.to(dtype)) for f in factors],
                   tensorized_shape, rank=rank)

    def __getitem__(self, indices):
        if not isinstance(indices, Iterable):
            indices = [indices]

        output_shape = []
        indexed_factors = []
        factors = self.factors
        weights = self.weights
        
        for (index, shape) in zip(indices, self.tensorized_shape):
            if isinstance(shape, int):
                # We are indexing a "regular" mode
                factor, *factors = factors
                
                if isinstance(index, (np.integer, int)):
                    weights = weights*factor[index, :]
                else:
                    factor = factor[index, :]
                    indexed_factors.append(factor)
                    output_shape.append(factor.shape[0])

            else: 
                # We are indexing a tensorized mode
                
                if index == slice(None) or index == ():
                    # Keeping all indices (:)
                    indexed_factors.extend(factors[:len(shape)])
                    output_shape.append(shape)

                else:
                    if isinstance(index, slice):
                        # Since we've already filtered out :, this is a partial slice
                        # Convert into list
                        max_index = math.prod(shape)
                        index = list(range(*index.indices(max_index)))

                    if isinstance(index, Iterable):
                        output_shape.append(len(index))

                    index = np.unravel_index(index, shape)
                    # Index the whole tensorized shape, resulting in a single factor
                    factor = 1
                    for idx, ff in zip(index, factors[:len(shape)]):
                        factor *= ff[idx, :]

                    if tl.ndim(factor) == 2:
                        indexed_factors.append(factor)
                    else:
                        weights = weights*factor

                factors = factors[len(shape):]
        
        indexed_factors.extend(factors)
        output_shape.extend(self.tensorized_shape[len(indices):])
        
        if indexed_factors:
            return self.__class__(weights, indexed_factors, tensorized_shape=output_shape)
        return tl.sum(weights)


class TuckerTensorized(TensorizedTensor, TuckerTensor, name='Tucker'):
    
    def __init__(self, core, factors, tensorized_shape, rank=None):
        tensor_shape = sum([(e,) if isinstance(e, int) else tuple(e) for e in tensorized_shape], ())

        super().__init__(core, factors, tensor_shape, rank)

        # Modify only what varies from the Tensor case
        self.shape = tensorized_shape_to_shape(tensorized_shape)
        self.tensorized_shape = tensorized_shape

    @classmethod
    def new(cls, tensorized_shape, rank, device=None, dtype=None, **kwargs):
        tensor_shape = sum([(e,) if isinstance(e, int) else tuple(e) for e in tensorized_shape], ())
        rank = tl.tucker_tensor.validate_tucker_rank(tensor_shape, rank)

        # Register the parameters
        core = nn.Parameter(torch.empty(rank, device=device, dtype=dtype))
        # Avoid the issues with ParameterList
        factors = [nn.Parameter(torch.empty((s, r), device=device, dtype=dtype)) for (s, r) in zip(tensor_shape, rank)]

        return cls(core, factors, tensorized_shape, rank=rank)

    @classmethod
    def from_tensor(cls, tensor, tensorized_shape, rank='same', fixed_rank_modes=None, **kwargs):
        shape = tensor.shape
        rank = tl.tucker_tensor.validate_tucker_rank(shape, rank, fixed_modes=fixed_rank_modes)

        with torch.no_grad():
            core, factors = tucker(tensor, rank, **kwargs)
        
        return cls(nn.Parameter(core.contiguous()), [nn.Parameter(f.contiguous()) for f in factors],
                   tensorized_shape, rank=rank)

    def __getitem__(self, indices):
        counter = 0
        ndim = self.core.ndim
        new_ndim = 0
        new_factors = []
        out_shape = []
        new_modes = []

        core = self.core
        
        for (index, shape) in zip(indices, self.tensorized_shape):
            if isinstance(shape, int):
                if index is Ellipsis:
                    raise ValueError(f'Ellipsis is not yet supported, yet got indices={indices}, indices[{i}]={index}.')
                factor = self.factors[counter]
                if isinstance(index, int):
                    core = tenalg.mode_dot(core, factor[index, :], new_ndim)
                else:
                    contracted = factor[index, :]
                    new_factors.append(contracted)
                    if contracted.shape[0] > 1:
                        out_shape.append(shape)
                        new_modes.append(new_ndim)
                        new_ndim += 1

                counter += 1

            else: # Tensorized dimension
                n_tensorized_modes = len(shape)

                if index == slice(None) or index == ():
                    new_factors.extend(self.factors[counter:counter+n_tensorized_modes])
                    out_shape.append(shape)
                    new_modes.extend([new_ndim+i for i in range(n_tensorized_modes)])
                    new_ndim += n_tensorized_modes

                else:
                    if isinstance(index, slice):
                        # Since we've already filtered out :, this is a partial slice
                        # Convert into list
                        max_index = math.prod(shape)
                        index = list(range(*index.indices(max_index)))
                    
                    index = np.unravel_index(index, shape)
                    
                    contraction_factors = [f[idx, :] for idx, f in zip(index, self.factors[counter:counter+n_tensorized_modes])]
                    if contraction_factors[0].ndim > 1:
                        shared_symbol = einsum_symbols[core.ndim+1]
                    else:
                        shared_symbol = ''
                    
                    core_symbols = ''.join(einsum_symbols[:core.ndim])
                    factors_symbols = ','.join([f'{shared_symbol}{s}' for s in core_symbols[new_ndim:new_ndim+n_tensorized_modes]])
                    res_symbol = core_symbols[:new_ndim] + shared_symbol + core_symbols[new_ndim+n_tensorized_modes:]
                    
                    if res_symbol:
                        eq =  core_symbols + ',' + factors_symbols + '->' + res_symbol
                    else:
                        eq =  core_symbols + ',' + factors_symbols
        
                    core = torch.einsum(eq, core, *contraction_factors)
                    
                    if contraction_factors[0].ndim > 1:
                        new_ndim += 1

                counter += n_tensorized_modes

        if counter <= ndim:
            out_shape.extend(list(core.shape[new_ndim:]))
            new_modes.extend(list(range(new_ndim, core.ndim)))
            new_factors.extend(self.factors[counter:])

        # Only here until our Tucker class handles partial-Tucker too
        if len(new_modes) != core.ndim:
            core = tenalg.multi_mode_dot(core, new_factors, new_modes)
            new_factors = []
        
        if new_factors:
            # return core, new_factors, out_shape, new_modes
            return self.__class__(core, new_factors, tensorized_shape=out_shape)

        return core

def validate_block_tt_rank(tensorized_shape, rank):
    ndim = max([1 if isinstance(s, int) else len(s) for s in tensorized_shape])
    factor_shapes = [(s, )*ndim if isinstance(s, int) else s for s in tensorized_shape]
    factor_shapes = list(math.prod(e) for e in zip(*factor_shapes))

    return tl.tt_tensor.validate_tt_rank(factor_shapes, rank)


class BlockTT(TensorizedTensor, name='BlockTT'):
    def __init__(self, factors, tensorized_shape=None, rank=None):
        super().__init__()
        self.shape = tensorized_shape_to_shape(tensorized_shape)
        self.tensorized_shape = tensorized_shape
        self.rank = rank
        self.order = len(self.shape)
        self.factors = FactorList(factors)

    @classmethod
    def new(cls, tensorized_shape, rank, device=None, dtype=None, **kwargs):
        if all(isinstance(s, int) for s in tensorized_shape):
            warnings.warn(f'Given a "flat" shape {tensorized_shape}. '
                          'This will be considered as the shape of a tensorized vector. '
                          'If you just want a 1D tensor, use a regular Tensor-Train. ')
            ndim = 1
            factor_shapes = [tensorized_shape]
            tensorized_shape = (tensorized_shape,)
        else:
            ndim = max([1 if isinstance(s, int) else len(s) for s in tensorized_shape])
            factor_shapes = [(s, )*ndim if isinstance(s, int) else s for s in tensorized_shape]
        
        rank = validate_block_tt_rank(tensorized_shape, rank)
        factor_shapes = [rank[:-1]] + factor_shapes + [rank[1:]]
        factor_shapes = list(zip(*factor_shapes))
        factors = [nn.Parameter(torch.empty(s, device=device, dtype=dtype)) for s in factor_shapes]
        
        return cls(factors, tensorized_shape=tensorized_shape, rank=rank)

    @property
    def decomposition(self):
        return self.factors

    def to_tensor(self):
        start = ord('d')
        in1_eq = []
        in2_eq = []
        out_eq = []
        for i, s in enumerate(self.tensorized_shape):
            in1_eq.append(start+i)
            if isinstance(s, int):
                in2_eq.append(start+i)
                out_eq.append(start+i)
            else:
                in2_eq.append(start+self.order+i)
                out_eq.append(start+i)
                out_eq.append(start+self.order+i)
        in1_eq = ''.join(chr(i) for i in in1_eq)
        in2_eq = ''.join(chr(i) for i in in2_eq)
        out_eq = ''.join(chr(i) for i in out_eq)
        equation = f'a{in1_eq}b,b{in2_eq}c->a{out_eq}c'
        
        for i, factor in enumerate(self.factors):
            if not i:
                res = factor
            else:
                out_shape = list(res.shape)
                for i, s in enumerate(self.tensorized_shape):
                    if not isinstance(s, int):
                        out_shape[i+1] *= factor.shape[i+1]
                out_shape[-1] = factor.shape[-1]
                res = tl.reshape(tl.einsum(equation, res, factor), out_shape)

        return tl.reshape(res.squeeze(0).squeeze(-1), self.tensor_shape)

    def __getitem__(self, indices):
        factors = self.factors
        if not isinstance(indices, Iterable):
            indices = [indices]

        if len(indices) < self.ndim:
            indices = list(indices)
            indices.extend([slice(None)]*(self.ndim - len(indices)))
        elif len(indices) > self.ndim:
            indices = [indices] # We're only indexing the first dimension

        output_shape = []
        indexed_factors = []
        ndim = len(self.factors)
        indexed_ndim = len(indices)

        contract_factors = False # If True, the result is dense, we need to form the full result
        contraction_op = [] # Whether the operation is batched or not
        eq_in1 = 'a' # Previously contracted factors (rank_0, dim_0, ..., dim_N, rank_k)
        eq_in2 = 'b' # Current factor (rank_k, dim_0', ..., dim_N', rank_{k+1})
        eq_out = 'a' # Output contracted factor (rank_0, dim_0", ..., dim_N", rank_{k_1})
        # where either:
        #     i. dim_k" = dim_k' = dim_k (contraction_op='b' for batched)
        # or ii. dim_k" = dim_k' x dim_k (contraction_op='m' for multiply)
        
        idx = ord('d') # Current character we can use for contraction
        
        pad = (slice(None), ) # index previous dimensions with [:], to avoid using .take(dim=k)
        add_pad = False       # whether to increment the padding post indexing
        
        for (index, shape) in zip(indices, self.tensorized_shape):
            if isinstance(shape, int):
                # We are indexing a "batched" mode, not a tensorized one            
                if not isinstance(index, (np.integer, int)):
                    if isinstance(index, slice):
                        index = list(range(*index.indices(shape)))

                    output_shape.append(len(index))
                    add_pad = True
                    contraction_op += 'b' # batched
                    eq_in1 += chr(idx); eq_in2 += chr(idx); eq_out += chr(idx)
                    idx += 1
                # else: we've essentially removed a mode of each factor
                index = [index]*ndim
            else: 
                # We are indexing a tensorized mode

                if index == slice(None) or index == ():
                    # Keeping all indices (:)
                    output_shape.append(shape)

                    eq_in1 += chr(idx)
                    eq_in2 += chr(idx+1)
                    eq_out += chr(idx) + chr(idx+1)
                    idx += 2
                    add_pad = True
                    index = [index]*ndim
                    contraction_op += 'm' # multiply
                else:
                    contract_factors = True

                    if isinstance(index, slice):
                        # Since we've already filtered out :, this is a partial slice
                        # Convert into list
                        max_index = math.prod(shape)
                        index = list(range(*index.indices(max_index)))

                    if isinstance(index, Iterable):
                        output_shape.append(len(index))
                        contraction_op += 'b' # multiply
                        eq_in1 += chr(idx)
                        eq_in2 += chr(idx)
                        eq_out += chr(idx)
                        idx += 1
                        add_pad = True

                    index = np.unravel_index(index, shape)

            # Index the whole tensorized shape, resulting in a single factor
            factors = [ff[pad + (idx,)] for (ff, idx) in zip(factors, index)]# + factors[indexed_ndim:]
            if add_pad:
                pad += (slice(None), )
                add_pad = False
                
#         output_shape.extend(self.tensorized_shape[indexed_ndim:])

        if contract_factors:
            eq_in2 += 'c'
            eq_in1 += 'b'
            eq_out += 'c'
            eq = eq_in1 + ',' + eq_in2 + '->' + eq_out
            for i, factor in enumerate(factors):
                if not i:
                    res = factor
                else:
                    out_shape = list(res.shape)
                    for j, s in enumerate(factor.shape[1:-1]):
                        if contraction_op[j] == 'm':
                            out_shape[j+1] *= s
                    out_shape[-1] = factor.shape[-1] # Last rank
                    res = tl.reshape(tl.einsum(eq, res, factor), out_shape)
            return res.squeeze()
        else:
            return self.__class__(factors, output_shape, self.rank)

    def normal_(self, mean=0, std=1):
        if mean != 0:
            raise ValueError(f'Currently only mean=0 is supported, but got mean={mean}')
            
        r = np.prod(self.rank)
        std_factors = (std/r)**(1/self.order)

        with torch.no_grad():
            for factor in self.factors:
                factor.data.normal_(0, std_factors)
        return self

    def __torch_function__(self, func, types, args=(), kwargs=None):
        if kwargs is None:
            kwargs = {}

        args = [t.to_matrix() if hasattr(t, 'to_matrix') else t for t in args]
        return func(*args, **kwargs)

    # def from_matrix(cls, matrix, tensorized_row_shape, tensorized_column_shape, rank, n_matrices=(), **kwargs):
    #     if matrix.ndim > 2:
    #         n_matrices = _ensure_tuple(tl.shape(matrix)[:-2])
    #     else:
    #         n_matrices = ()
    #     tensor = matrix.reshape((*n_matrices, *tensorized_row_shape, *tensorized_column_shape))
    @classmethod
    def from_tensor(cls, tensor, tensorized_shape, rank, **kwargs):
        rank = tl.tt_matrix.validate_tt_matrix_rank(tensor.shape, rank)

        with torch.no_grad():
            factors = tensor_train_matrix(tensor, rank, **kwargs)
        factors = [nn.Parameter(f) for f in factors]

        return cls(factors, tensorized_shape, rank)

    def init_from_tensor(self, tensor, **kwargs):
        rank = tl.tt_matrix.validate_tt_matrix_rank(tensor.shape, self.rank)

        with torch.no_grad():
            factors = tensor_train_matrix(tensor, rank, **kwargs)

        self.factors = FactorList([nn.Parameter(f) for f in factors])
        self.rank = tuple([f.shape[0] for f in factors] + [1])

        return self




def validate_block_cp_rank(tensor_shape, rank='same', rounding='round'):
    """Returns the rank of a BlockCP Decomposition

    Parameters
    ----------
    tensor_shape : tupe
        shape of the tensor to decompose
    rank : {'same', float, int}, default is same
        way to determine the rank, by default 'same'
        if 'same': rank is computed to keep the number of parameters (at most) the same
        if float, computes a rank so as to keep rank percent of the original number of parameters
        if int, just returns rank
    rounding = {'round', 'floor', 'ceil'}

    Returns
    -------
    rank : int
        rank of the decomposition
    """
    # print("TENSOR SHAPE:", tensor_shape)

    if is_tensorized_shape(tensor_shape):
        tensor_shape = tensorized_shape_to_shape(tensor_shape)
    if rounding == 'ceil':
        rounding_fun = np.ceil
    elif rounding == 'floor':
        rounding_fun = np.floor
    elif rounding == 'round':
        rounding_fun = np.round
    else:
        raise ValueError(f'Rounding should be of round, floor or ceil, but got {rounding}')
    
    if rank == 'same':
        rank = float(1)

    if isinstance(rank, float):
        rank = int(rounding_fun(np.prod(tensor_shape)*rank/np.sum(tensor_shape)))
    return rank

class BlockCP(TensorizedTensor, name='BlockCP'):
    """BlockCP Factorization

    Parameters
    ----------
    weights
    factors
    shape
    rank
    """

    def __init__(self, weights, factors, tensorized_shape=None, rank=None):
        super().__init__()
        self.shape = tensorized_shape_to_shape(tensorized_shape)
        self.tensorized_shape = tensorized_shape
        self.rank = rank
        self.order = len(self.shape)
        self.weights = weights
        self.factors = FactorList(factors)
    
    @classmethod
    def new(cls, tensorized_shape, rank, device=None, dtype=None, **kwargs):

        if all(isinstance(s,int) for s in tensorized_shape):
            warnings.warn(f'Given a "flat" shape {tensorized_shape}, '
                          ' This will be considered as the shape of a tensorized vector. '
                          ' If you just want a 1D tensor, used CP')
            ndim = 1
            factor_shapes = [tensorized_shape]
            tensorized_shape = (tensorized_shape,)

        else:
            #*
            ndim = max([1 if isinstance(s, int) else len(s) for s in tensorized_shape])

            #*
            factor_shapes = [(s, )*ndim if isinstance(s, int) else s for s in tensorized_shape]

        #  changed from shape to tensorized_shape

        rank = validate_block_cp_rank(tensorized_shape, rank)
        # # Register the parameters
        weights = nn.Parameter(torch.empty(rank, device = device, dtype = dtype))

        ranks = [rank] * len(tensorized_shape[1])
        factor_shapes = factor_shapes + [ranks]
        factor_shapes = list(zip(*factor_shapes))
        factors = [nn.Parameter(torch.empty(s)) for s in factor_shapes]

        return cls(weights, factors, tensorized_shape, rank=rank)


    @classmethod
    def from_tensor(cls, tensor, tensorized_shape, rank='same', **kwargs):
        
        """
        Note: Not Implemented because we need a version of parafac 
              for order-n factors, currently have for order 2 (shape, rank)
        """
        
        raise NotImplementedError("Not Implemented because we need a version of " + 
                                  "parafac for order-n factors, currently have for order 2 (shape, rank) ")

        shape = tensor.shape
        rank = bct.validate_block_cp_rank(shape, rank)
        dtype = tensor.dtype

        with torch.no_grad():
            weights, factors = parafac(tensor.to(torch.float64), rank, **kwargs)
        
        return cls(nn.Parameter(weights.to(dtype)), [nn.Parameter(f.to(dtype)) for f in factors])


    def init_from_tensor(self, tensor, l2_reg=1e-5, **kwargs):
        
        """
        Note: Not Implemented because we need a version of parafac 
              for order-n factors, currently have for order 2 (shape, rank)
        """
        
        raise NotImplementedError("Not Implemented because we need a version of " + 
                                  "parafac for order-n factors, currently have for order 2 (shape, rank) ")
            
        with torch.no_grad():
            weights, factors = parafac(tensor, self.rank, l2_reg=l2_reg, **kwargs)
        
        self.weights = nn.Parameter(weights)
        self.factors = FactorList([nn.Parameter(f) for f in factors])
        return self

    @property
    def decomposition(self):
        return self.weights, self.factors

    def to_tensor(self):
        factors = self.factors
        weights = self.weights
        ndim = len(factors)

        rank_ind = ord('a')
        
        if isinstance(self.tensorized_shape[0], int):
            batched = True
            batch_size  = self.tensorized_shape[0]
        else:
            batched = False

        if batched:
            batch_ind = ord('b')
        idx = ord('c') # Current character we can use for contraction
        eq_terms = {}
        for i in range(ndim):
            eq_terms[i] = chr(rank_ind)
        for i in range(ndim):
            chr_idx = chr(idx)
            idx +=1
            chr_idx2 = chr(idx)
            eq_terms[i] = chr_idx + chr_idx2 + chr(rank_ind)
            if batched:
                eq_terms[i] = chr(batch_ind) + eq_terms[i]
            idx +=1
        eq_out = ''
        if batched:
            eq_out += chr(batch_ind)
        for j in range(1, 3):
            if batched:
                eq_out += ''.join([eq_terms[i][j] for i in range(len(eq_terms.keys()))])
            else:
                eq_out += ''.join([eq_terms[i][j-1] for i in range(len(eq_terms.keys()))])
        eq_in = ','.join([*eq_terms.values()])
        eq = eq_in + ',' + chr(rank_ind) + '->' + eq_out

        res = tl.einsum(eq, *self.factors, weights)
        return tl.reshape(res, self.tensor_shape)
    


    def normal_(self, mean=0, std=1):
        super().normal_(mean, std)
        std_factors = (std/math.sqrt(self.rank))**(1/self.order)

        with torch.no_grad():
            self.weights.fill_(1)
            for factor in self.factors:
                factor.data.normal_(0, std_factors)
        return self


    def __torch_function__(self, func, types, args=(), kwargs=None):
        if kwargs is None:
            kwargs = {}

        args = [t.to_matrix() if hasattr(t, 'to_matrix') else t for t in args]
        return func(*args, **kwargs)


    def __getitem__(self, indices):
        factors = self.factors
        weights = self.weights
        if not isinstance(indices, Iterable):
            indices = [indices]

        if len(indices) < self.ndim:
            indices = list(indices)
            (indices.extend([slice(None)]*(self.ndim - len(indices)))) ### e.g. if you index [0], goes to [0, slice(None, None, None)]

        elif len(indices) > self.ndim:
            indices = [indices] # We're only indexing the first dimension
        
        if isinstance(self.tensorized_shape[0], int):
            batched = True
            batch_size = self.tensorized_shape[0]
        else:
            batched = False

        output_shape = []
        ndim = len(self.factors)

        contract_factors = False # If True, the result is dense, we need to form the full result
        rank_ind = ord('a') # BlockCP rank character
        if batched:
            batch_ind = ord('b')  # batch character

        idx = ord('c') # Current character we can use for contraction
        eq_terms = {}
        for i in range(ndim):
            eq_terms[i] = chr(rank_ind)
        eq_out = ''
        
        pad = ()
        add_pad = False       # whether to increment the padding post indexing
        rebatched = False     # Add a dim for batch size if only one item has been selected

        m = 0
        for (index, shape) in zip(indices, self.tensorized_shape):
            if isinstance(shape, int):
                # We are indexing a "batched" mode, not a tensorized one            
                if not isinstance(index, (np.integer, int)):
                    if isinstance(index, slice):
                        index = list(range(*index.indices(shape)))
                        batch_size = len(index)

                    output_shape.append(len(index))
                    add_pad = True

                # else: we've essentially removed a mode of each factor
                else:
                    batch_size = 1
                index = [index]*ndim
            else: 
                # We are indexing a tensorized mode
                
                if index == slice(None) or index == ():
                    # Keeping all indices (:)
                    output_shape.append(shape)
                    for i in range(ndim):
                        chr_idx = chr(idx)
                        eq_terms[i] = chr_idx + eq_terms[i] #+ chr(rank_ind)
                        idx +=1
                    eq_out += ''.join([eq_terms[i][0] for i in range(len(eq_terms.keys()))])
                    add_pad = True
                    index = [index]*ndim
                else:

                    ## index is an integer
                    contract_factors = True

                    if isinstance(index, slice):
                        # Since we've already filtered out :, this is a partial slice
                        # Convert into list
                        max_index = math.prod(shape)
                        index = list(range(*index.indices(max_index)))
                        add_pad = True

                    if isinstance(index, Iterable):
                        output_shape.append(len(index))
                        for i in range(ndim):
                            chr_idx = chr(idx)
                            eq_terms[i] = chr_idx + eq_terms[i] 
                        eq_out += chr(idx)
                        idx += 1
                        add_pad = True
                    
                    index = np.unravel_index(index, shape)
          
            factors = [ff[pad + (idx,)] for (ff, idx) in zip(factors, index)]# + factors[indexed_ndim:]
            if add_pad:
                pad += (slice(None), )
                add_pad = False

        if contract_factors:
            if batched and batch_size != 1: # only append a batch dimension if that batch_size > 1, batch_size = 1 has no batch dimension
                for i in range(ndim):
                    eq_terms[i] = chr(batch_ind) + eq_terms[i]
                eq_out =  chr(batch_ind) + eq_out
            eq_in = ','.join([*eq_terms.values()])
            eq = eq_in + ',' + chr(rank_ind) +  '->' + eq_out
            for i in range(len(factors)):
                if all( isinstance(ind, slice) for ind in indices):
                    factors[i] = factors[i].transpose(1,0)
            res = tl.einsum(eq, *factors, weights)

            if not batched:
                if any(isinstance(x, int) for x in indices ) and not all(isinstance(x, int) for x in indices ):
                    res = res.flatten()
            else:
                if any(isinstance(x, int) for x in indices[1:] ) and not all(isinstance(x, int) for x in indices[1:] ): 
                    res = res.reshape(batch_size, -1).squeeze(0)
        
            # ensure correct shape for when a single nontrivial slices is taken
            output_shape = [s if isinstance(s, int) else np.prod(s) for s in output_shape]
            res= res.reshape(output_shape)
            
            return res
        else:
            return self.__class__(self.weights, factors, output_shape, self.rank)
