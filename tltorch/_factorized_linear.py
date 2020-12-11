"""Tensor Regression Layers
"""

# Author: Jean Kossaifi
# License: BSD 3 clause

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

import tensorly as tl
tl.set_backend('pytorch')
from tensorly import tenalg
from tensorly.random import random_tucker, random_cp, random_tt, random_tt_matrix
from tensorly.decomposition import parafac, tucker, tensor_train, tensor_train_matrix
from tensorly import random
from tensorly import testing
from tensorly import (validate_tt_rank, validate_cp_rank, 
                      validate_tucker_rank, validate_tt_matrix_rank)

from .base import TensorModule
from . import init


class BaseFactorizedLinear(TensorModule):
    """Tensorized Fully-Connected Layers

        The weight matrice is tensorized to a tensor of size `tensorized_shape`.
        That tensor is expressed as a low-rank tensor.
        During inference, the full tensor is reconstructed, and unfolded back into a matrix, 
        used for the forward pass in a regular linear layer.

    Parameters
    ----------
    in_features : int
    out_features : int
    tensorized_shape : int tuple
    rank : int tuple or str
    bias : bool, default is True
    """
    def __init__(self, in_features, out_features, tensorized_shape, rank, bias=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.tensorized_shape = tensorized_shape
        self.weight_shape = (out_features, in_features)
        self.rank = rank
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)

    @classmethod
    def from_linear(cls, linear, tensorized_shape, rank, bias=True):
        """Class method to create an instance from an existing linear layer

        Parameters
        ----------
        linear : torch.nn.Linear
            layer to tensorize
        tensorized_shape : tuple
            shape to tensorized the weight matrix to.
            Must verify np.prod(tensorized_shape) == np.prod(linear.weight.shape)
        rank :  {rank of the decomposition, 'same', float}
            if float, percentage of parameters of the original weights to use
            if 'same' use the same number of parameters
        bias : bool, default is True
        """
        out_features, in_features = linear.weight.shape
        instance = cls(in_features, out_features, tensorized_shape=tensorized_shape, rank=rank, bias=bias)
        instance.init_from_tensor(linear.weight, linear.bias)
        return instance

    def __repr__(self):
        msg = f'Factorized {self.__class__.__name__} with {self.in_features} inputs and {self.out_features} outputs.\n'
        msg += f'  Weight ({self.out_features}, {self.in_features}) tensorized to {self.tensorized_shape} with rank {self.rank}.'
        return msg 

    def __getattr__(self, name):
        """Hack for PyTorch to be able to use the full reconstructed weight in attention layers

        Simply defining a property `weight` is not sufficient as PyTorch will first look 
        in self._parameters
        """
        if name == 'weight':
            return self.full_weight
        else:
            return super().__getattr__(name)

    
class TuckerLinear(BaseFactorizedLinear):
    """Tensorized Fully-Connected Layers

        The weight matrice is tensorized to a tensor of size `tensorized_shape`.
        That tensor is expressed as a low-rank (Tucker) tensor.
        During inference, the full tensor is reconstructed, and unfolded back into a matrix, 
        used for the forward pass in a regular linear layer.

    Parameters
    ----------
    in_features : int
    out_features : int
    tensorized_shape : int tuple
    rank : int tuple or str
    bias : bool, default is True

    See also
    --------
    TTLinear
    CPLinear
    """
    def __init__(self, in_features, out_features, tensorized_shape, rank, bias=True):
        super().__init__(in_features, out_features, tensorized_shape, rank, bias=bias)
        self.rank = validate_tucker_rank(tensorized_shape, rank=rank)           

        self.core = nn.Parameter(torch.Tensor(*self.rank))
        self.factors = nn.ParameterList(nn.Parameter(torch.Tensor(s, r))\
                                        for (s, r) in zip(tensorized_shape, self.rank))

        self.init_from_random(False)

    def forward(self, input):
        """Inference using the tensorized and factorized weight matrix"""
        weight = tl.tucker_to_tensor(self._process_decomposition()).reshape(self.weight_shape)
        
        return F.linear(input, weight, self.bias)

    def init_from_random(self, decompose_full_weight=True):
        """Initializes the factorization randomly

        Parameters
        ----------
        decompose_full_weight : bool, default is True
            if True, a full weight is created and decomposed to initialize the factors
            otherwise, the factors of the decomposition are directly initialized
        """
        if decompose_full_weight:
            full_weight = torch.normal(0.0, 0.02, size=self.weight_shape)
            self.init_from_tensor(full_weight)
        else:
            init.tucker_init(self.core, self.factors)
        if self.bias is not None:
            self.bias.data.zero_()

    def init_from_decomposition(self, tucker_tensor, bias=None):
        """Initializes the factorization from the given decomposition

        Parameters
        ----------
        tucker_tensor : (core, factors)
            values to initialize the decomposition parametrizing the layer to
        bias : torch.Tensor or None, default is None
        """
        core, factors = tucker_tensor
        with torch.no_grad():
            for i, f in enumerate(factors):
                self.factors[i].data = f
            self.core.data = core

        if self.bias is not None and bias is not None:
            self.bias.data = bias

    def init_from_tensor(self, tensor, bias=None, decomposition_kwargs=None):
        """Initializes the layer by decomposing a full tensor

        Parameters
        ----------
        tensor : torch.Tensor
            must be either a matrix or a tensor
            must verify ``np.prod(tensor.shape) == np.prod(self.tensorized_shape)``
        bias : torch.Tensor or None, default is None
        decomposition_kwargs : dict
            dictionary of parameters passed directly to TensorLy for the decomposition
        """
        with torch.no_grad():
            tensor = tensor.reshape(self.tensorized_shape)
            tucker_tensor = tucker(tensor, rank=self.rank, init='random')
        self.init_from_decomposition(tucker_tensor, bias=bias)

    @property
    def full_weight(self):
        """Returns the reconstruced matrix weight of the linear layer
        """
        return tl.reshape(tl.tucker_to_tensor((self.core, self.factors)), self.weight_shape)

    def get_decomposition(self):
        """Returns the decomposition parametrizing the layer
        """
        return self.core, self.factors

class CPLinear(BaseFactorizedLinear):
    """Tensorized Fully-Connected Layers

        The weight matrice is tensorized to a tensor of size `tensorized_shape`.
        That tensor is expressed as a low-rank (CP) tensor.
        During inference, the full tensor is reconstructed, and unfolded back into a matrix, 
        used for the forward pass in a regular linear layer.

    Parameters
    ----------
    in_features : int
    out_features : int
    tensorized_shape : int tuple
    rank : int tuple or str
    bias : bool, default is True

    See also
    --------
    TTLinear
    TuckerLinear
    """
    def __init__(self, in_features, out_features, tensorized_shape, rank, bias=True):
        super().__init__(in_features, out_features, tensorized_shape, rank, bias=bias)
        self.rank = validate_cp_rank(tensorized_shape, rank=rank)           

        self.weights = nn.Parameter(torch.Tensor(self.rank))
        self.factors = nn.ParameterList(nn.Parameter(torch.Tensor(s, self.rank)) for s in tensorized_shape)

        self.init_from_random(decompose_full_weight=False)

    def forward(self, input):
        """Inference using the tensorized and factorized weight matrix"""
        weight = tl.cp_to_tensor(self._process_decomposition()).reshape(self.weight_shape)
        
        return F.linear(input, weight, self.bias)

    def init_from_random(self, decompose_full_weight=True):
        """Initializes the factorization randomly

        Parameters
        ----------
        decompose_full_weight : bool, default is True
            if True, a full weight is created and decomposed to initialize the factors
            otherwise, the factors of the decomposition are directly initialized
        """
        if decompose_full_weight:
            full_weight = torch.normal(0.0, 0.02, size=self.weight_shape)
            self.init_from_tensor(full_weight)
        else:
            init.cp_init(self.weights, self.factors)
        if self.bias is not None:
            self.bias.data.zero_()

    def init_from_decomposition(self, cp_tensor, bias=None):
        """Initializes the factorization from the given decomposition

        Parameters
        ----------
        tucker_tensor : (weights, factors)
            values to initialize the decomposition parametrizing the layer to
        bias : torch.Tensor or None, default is None
        """
        weights, factors = cp_tensor
        with torch.no_grad():
            for i, f in enumerate(factors):
                self.factors[i].data = f
            self.weights.data = weights
            
            if self.bias is not None and bias is not None:
                self.bias.data = bias

    def init_from_tensor(self, tensor, bias=None, decomposition_kwargs=dict(init='random')):
        """Initializes the layer by decomposing a full tensor

        Parameters
        ----------
        tensor : torch.Tensor
            must be either a matrix or a tensor
            must verify ``np.prod(tensor.shape) == np.prod(self.tensorized_shape)``
        bias : torch.Tensor or None, default is None
        """
        with torch.no_grad():
            tensor = tensor.reshape(self.tensorized_shape)
            print(tensor.shape)
            cp_tensor = parafac(tensor, rank=self.rank, **decomposition_kwargs)
        self.init_from_decomposition(cp_tensor, bias=bias)

    def get_decomposition(self):
        """Returns the decomposition parametrizing the layer
        """
        return (self.weights, self.factors)

    @property
    def full_weight(self):
        """Returns the reconstruced matrix weight of the linear layer
        """
        return tl.reshape(tl.cp_to_tensor((self.weights, self.factors)), self.weight_shape)


class TTLinear(BaseFactorizedLinear):
    """Tensorized Fully-Connected Layers

        The weight matrice is tensorized to a tensor of size `tensorized_shape`.
        That tensor is expressed as a low-rank (TT) tensor.
        During inference, the full tensor is reconstructed, and unfolded back into a matrix, 
        used for the forward pass in a regular linear layer.

    Parameters
    ----------
    in_features : int
    out_features : int
    tensorized_shape : int tuple
    rank : int tuple or str
    bias : bool, default is True

    See also
    --------
    TuckerLinear
    CPLinear
    TTMLinear
    
    Notes
    -----
    This is very similar to [1]_ except that the weight matrix is simply **reshaped** into a tensor,
    while in [1]_, the dimensions are then also permuted in order to jointly compress input and outputs.
    The original [1]_ is implemented in :func:`tltorch.TTMLinear` .

    References
    ----------
    .. [1] Tensorizing Neural Networks, Alexander Novikov, Dmitry Podoprikhin, Anton Osokin, Dmitry Vetrov
    """
    def __init__(self, in_features, out_features, tensorized_shape, rank, bias=True):
        super().__init__(in_features, out_features, tensorized_shape, rank, bias=bias)
        self.rank = validate_tt_rank(tensorized_shape, rank=rank)           
        self.factors = nn.ParameterList()
        for i, s in enumerate(self.tensorized_shape):
            self.factors.append(nn.Parameter(torch.Tensor(self.rank[i], s, self.rank[i+1])))

        # Things like setting the tt_shape above are the init is not in the base class
        self.init_from_random(decompose_full_weight=False)

    def forward(self, input):
        """Inference using the tensorized and factorized weight matrix"""
        weight = tl.tt_to_tensor(self._process_decomposition()).reshape(self.weight_shape)
        
        return F.linear(input, weight, self.bias)

    def init_from_random(self, decompose_full_weight=True):
        """Initializes the factorization randomly

        Parameters
        ----------
        decompose_full_weight : bool, default is True
            if True, a full weight is created and decomposed to initialize the factors
            otherwise, the factors of the decomposition are directly initialized
        """
        if decompose_full_weight:
            full_weight = torch.normal(0.0, 0.02, size=self.tensorized_shape)
            self.init_from_tensor(full_weight)
        else:
            init.tt_init(self.factors)
        if self.bias is not None:
            self.bias.data.zero_()

    def init_from_decomposition(self, tt_tensor, bias=None):
        """Initializes the factorization from the given decomposition

        Parameters
        ----------
        tucker_tensor : (factors)
            values to initialize the decomposition parametrizing the layer to
        bias : torch.Tensor or None, default is None
        """
        factors = tt_tensor
        for i, factor in enumerate(factors):
            self.factors[i].data = factor

        if self.bias is not None and bias is not None:
            self.bias.data = bias

    def init_from_tensor(self, tensor, bias=None, decomposition_kwargs=dict()):
        """Initializes the layer by decomposing a full tensor

        Parameters
        ----------
        tensor : torch.Tensor
            must be either a matrix or a tensor
            must verify ``np.prod(tensor.shape) == np.prod(self.tensorized_shape)``
        bias : torch.Tensor or None, default is None
        """
        with torch.no_grad():
            tensor = tensor.reshape(self.tensorized_shape)
            tt_tensor = tensor_train(tensor, rank=self.rank, **decomposition_kwargs)
        self.init_from_decomposition(tt_tensor, bias=bias)

    def get_decomposition(self):
        """Returns the decomposition parametrizing the layer
        """
        return self.factors

    @property
    def full_weight(self):
        """Returns the reconstruced matrix weight of the linear layer
        """
        return tl.reshape(tl.tt_to_tensor(self.factors), self.weight_shape)


class TTMLinear(BaseFactorizedLinear):
    """Tensorized Fully-Connected Layers in the TT-Matrix format [1]_

        The weight matrice is tensorized to a tensor of size `tensorized_shape`.
        That tensor is expressed as a low-rank TT-Matrix by jointly compressing inputs and outputs.
        During inference, the full tensor is reconstructed, and unfolded back into a matrix, 
        used for the forward pass in a regular linear layer.

    Parameters
    ----------
    in_features : int
    out_features : int
    tensorized_shape : int tuple
        should be left_shape + right_shape correponsding to a weight matrix of size left x right
    rank : int tuple or str, default is 'same'
    bias : bool, default is True

    See also
    --------
    TuckerLinear
    CPLinear
    TTLinear
    
    Notes
    -----
    This layer permutes the dimensions of the weight matrix after it has been reshaped into 
    a higher-order tensor of shape `tensorized_shape`.

    For a linear layer with `out_feature = O_1 * O_2 * O_3` and `in_features = I_1 * I_2 * I_3`
    and `tensorized_shape = (O_1, O_2, O_3, I_1, I_2, I_3)` and rank `R = (R_1, R_2, R_3, R_4)` with `R_1 = R_4 = 1`.
    the inputs and outputs will be jointly compressed by each tt-matrix core.
    In other words, the k-th core will be of shape `(R_k, O_k, I_k, R_{k+1})`.

    By contrast, :func:`tltorch.TTLinear` simply reshapes the matrix to `tensorized_shape` 
    and compresses with a tensor-train decomposition.

    References
    ----------
    .. [1] Tensorizing Neural Networks, Alexander Novikov, Dmitry Podoprikhin, Anton Osokin, Dmitry Vetrov
    """
    def __init__(self, in_features, out_features, tensorized_shape, rank='same', bias=True):
        super().__init__(in_features, out_features, tensorized_shape, rank, bias=bias)
        self.rank = validate_tt_matrix_rank(tensorized_shape, rank=rank)           
        self.factors = nn.ParameterList()
        self.ndim = len(tensorized_shape) // 2
        self.out_shape = tensorized_shape[:self.ndim]
        self.in_shape = tensorized_shape[self.ndim:]

        for i, (s_out, s_in) in enumerate(zip(self.out_shape, self.in_shape)):
            self.factors.append(nn.Parameter(torch.Tensor(self.rank[i], s_out, s_in, self.rank[i+1])))

        # Things like setting the tt_shape above are the init is not in the base class
        self.init_from_random(decompose_full_weight=False)

    def forward(self, input):
        """Inference using the tensorized and factorized weight matrix"""
        weight = tl.tt_matrix_to_tensor(self._process_decomposition()).reshape(self.weight_shape)
        
        return F.linear(input, weight, self.bias)

    def init_from_random(self, decompose_full_weight=True):
        """Initializes the factorization randomly

        Parameters
        ----------
        decompose_full_weight : bool, default is True
            if True, a full weight is created and decomposed to initialize the factors
            otherwise, the factors of the decomposition are directly initialized
        """
        if decompose_full_weight:
            full_weight = torch.normal(0.0, 0.02, size=self.tensorized_shape)
            self.init_from_tensor(full_weight)
        else:
            init.tt_matrix_init(self.factors)
        if self.bias is not None:
            self.bias.data.zero_()

    def init_from_decomposition(self, tt_matrix, bias=None):
        """Initializes the factorization from the given decomposition

        Parameters
        ----------
        tucker_tensor : (factors)
            values to initialize the decomposition parametrizing the layer to
        bias : torch.Tensor or None, default is None
        """
        factors = tt_matrix
        for i, factor in enumerate(factors):
            self.factors[i].data = factor

        if self.bias is not None and bias is not None:
            self.bias.data = bias

    def init_from_tensor(self, tensor, bias=None):
        """Initializes the layer by decomposing a full tensor

        Parameters
        ----------
        tensor : torch.Tensor
            must be either a matrix or a tensor
            must verify ``np.prod(tensor.shape) == np.prod(self.tensorized_shape)``
        bias : torch.Tensor or None, default is None
        """
        with torch.no_grad():
            tensor = tensor.reshape(self.tensorized_shape)
            tt_matrix = tensor_train_matrix(tensor, rank=self.rank)
        self.init_from_decomposition(tt_matrix, bias=bias)

    def get_decomposition(self):
        """Returns the decomposition parametrizing the layer
        """
        return self.factors

    @property
    def full_weight(self):
        """Returns the reconstruced matrix weight of the linear layer
        """
        return tl.reshape(tl.tt_matrix_to_tensor(self.factors), self.weight_shape)
