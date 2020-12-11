"""Tensor Regression Layers
"""

# Author: Jean Kossaifi
# License: BSD 3 clause

import math
import torch
import torch.nn as nn

import tensorly as tl
tl.set_backend('pytorch')
from tensorly import tenalg
from tensorly.random import random_tucker, random_cp, random_tt
from tensorly.decomposition import parafac, tucker, tensor_train
from tensorly import validate_tt_rank, validate_cp_rank, validate_tucker_rank

from .base import TensorModule
from . import init

class BaseTRL(TensorModule):
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
    def __init__(self, input_shape, output_shape, bias=False, verbose=0, **kwargs):
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

    def forward(self, x):
        """Performs a forward pass"""
        raise NotImplementedError
    
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
            self.init_from_tensor(full_weight)
        else:
            raise NotImplementedError()

    def init_from_decomposition(self, bias=None):
        """Initializes the factorization from the given decomposition

        Parameters
        ----------
        decomposed_tensor
            values to initialize the decomposition parametrizing the layer to
        bias : torch.Tensor or None, default is None
        """
        raise NotImplementedError()

    def init_from_tensor(self, tensor, bias=None, decomposition_kwargs=dict()):
        """Initializes the layer by decomposing a full tensor

        Parameters
        ----------
        tensor : torch.Tensor
            must be either a matrix or a tensor
            must verify ``np.prod(tensor.shape) == np.prod(self.tensorized_shape)``
        bias : torch.Tensor or None, default is None
        decomposition_kwargs : dict
            optional dictionary of parameters to pass to the decomposition
        """
        raise NotImplementedError()

    def get_decomposition(self):
        """Returns the decomposition parametrizing the layer
        """
        raise NotImplementedError()

class TuckerTRL(BaseTRL):
    """Tensor Regression Layer with Tucker weights [1]_

    Parameters
    ----------
    input_shape : int iterable
        shape of the input, excluding batch size
    output_shape : int iterable
        shape of the output, excluding batch size
    rank : int or int list
        rank of the Tucker weights
        if int, the same rank will be used for all dimensions
    project_input : bool, default is False
        is True, the input activations are first projected
        using factors from the low-rank Tucker weights
    verbose : int, default is 0
        level of verbosity

    See Also
    --------
    CPTRL
    TensorTrainTRL

    References
    ----------
    .. [1] Tensor Regression Networks, Jean Kossaifi, Zachary C. Lipton, Arinbjorn Kolbeinsson, 
        Aran Khanna, Tommaso Furlanello, Anima Anandkumar, JMLR, 2020. 
    """
    def __init__(self, input_shape, output_shape, rank, project_input=False,
                  bias=False, verbose=0, **kwargs):
        super().__init__(input_shape, output_shape, bias=bias, verbose=verbose, **kwargs)

        self.order = len(self.weight_shape)
        self.rank = validate_tucker_rank(self.weight_shape, rank=rank)           
                
        # Start at 1 as the batch-size is not projected
        self.projection_modes_input = tuple(range(1, self.n_input+1))
        # Start at 0 as weights don't have batch-size mode
        self.projection_modes_weights = tuple(range(self.n_input, self.n_input+self.n_output))
        self.project_input = project_input

        self.core = nn.Parameter(torch.Tensor(*self.rank))
        self.factors = nn.ParameterList(nn.Parameter(torch.Tensor(s, r))\
                                        for (s, r) in zip(self.weight_shape, self.rank))

        self.n_factor = len(self.factors)
        
        self.init_from_random(decompose_full_weight=False)

    def forward(self, x):
        core, factors = self._process_decomposition()

        if self.project_input:
            x = tenalg.multi_mode_dot(x, [factors[i] for i in range(self.n_input)], modes=self.projection_modes_input, transpose=True)
            regression_weights = tenalg.multi_mode_dot(core, [factors[i] for i in range(self.n_input, self.n_factor)], modes=self.projection_modes_weights)
        else:
            regression_weights = tl.tucker_to_tensor((core, factors))

        if self.bias is None:
            return tenalg.inner(x, regression_weights, n_modes=tl.ndim(x)-1)
        else:
            return tenalg.inner(x, regression_weights, n_modes=tl.ndim(x)-1) + self.bias
    
    def init_from_random(self, decompose_full_weight=False):
        if decompose_full_weight:
            full_weight = torch.normal(0.0, 0.02, size=self.weight_shape)
            self.init_from_tensor(full_weight)
        else:
            init.tucker_init(self.core, self.factors)
        if self.bias is not None:
            self.bias.data.zero_()

    def init_from_decomposition(self, tucker_tensor, bias=None):
        core, factors = tucker_tensor
        with torch.no_grad():
            for i, f in enumerate(factors):
                self.factors[i].data = f
            self.core.data = core

        if self.bias is not None and bias is not None:
            self.bias.data = bias.view(self.output_shape)

    def init_from_tensor(self, tensor, bias=None, decomposition_kwargs=dict(init='random')):
        with torch.no_grad():
            tucker_tensor = tucker(tensor, rank=self.rank, **decomposition_kwargs)
        self.init_from_decomposition(tucker_tensor, bias=bias)
        
    def init_from_linear(self, weight, bias, pooling_modes=None):                                                                                                                                                                                                                                                                       
        """Initialise the TRL from the weights of a fully connected layer
        """
        if pooling_modes is not None:
            pooling_modes = sorted(pooling_modes)
            weight_shape = list(self.weight_shape)
            for mode in pooling_modes[::-1]:
                if self.rank[mode] != 1:
                    msg = 'When initializing from a Fully-Connected layer,'
                    msg += ' it is only possible to learn pooling with a rank of 1.'
                    msg += f'However, got pooling_modes={pooling_modes} but rank[{mode}] = 1.'
                    raise ValueError(msg)
                if mode == 0:
                    raise ValueError(f'Cannot learn pooling for mode-0 (channels).')
                if mode > self.n_input:
                    msg = 'Can only learn pooling for the input tensor. '
                    msg += f'The input has only {self.n_input} modes, yet got a pooling on mode {mode}.'
                    raise ValueError(msg)

                weight_shape.pop(mode)
                        
            rank = tuple(r for (i, r) in enumerate(self.rank) if i not in pooling_modes)

            if self.verbose:
                print(f' Initializing {pooling_modes} with average pooling.')
        else:
            weight_shape = self.weight_shape
            rank = self.rank
        
        weight = torch.t(weight).contiguous().view(weight_shape)

        if self.verbose:
            print(f'Initializing TuckerTRL from linear weight of shape {weight.shape} with rank-{rank} Tucker decomposition.')
            
        core, factors = tucker(weight.data, rank=rank, n_iter_max=10, verbose=self.verbose)

        if pooling_modes is not None:
            # Initialise with average pooling
            for mode in pooling_modes:
                size = self.weight_shape[mode]
                factor = torch.ones(size, 1)/size
                factors.insert(mode, factor)
                core = core.unsqueeze(mode)
        
        self.init_from_decomposition((core, factors), bias=bias)

        if self.verbose:
            print('TRL successfully initialized.')

    def full_weight(self):
        """Return the reconstructed weights from the low_rank

        Returns
        -------
        tensor :
            weights recoonstructed from the low-rank ones learnt
        """
        return tl.tucker_to_tensor((self.core, self.factors))

    def get_decomposition(self):
        return (self.core, self.factors)


class CPTRL(BaseTRL):
    """Tensor Regression Layer with CP weights [1]_, [2]_
    
    Parameters
    -----------
    input_shape : int iterable
        shape of the input, excluding batch size
    output_shape : int iterable
        shape of the output, excluding batch size
    verbose : int, default is 0
        level of verbosity
    rank : int 
        rank of the CP weights
    verbose : int, default is 1
        level of verbosity, if 0, no information will be printed
    
    See Also
    --------
    TuckerTRL
    TensorTrainTRL

    References
    ----------
    .. [1] Tensor Regression Networks, Jean Kossaifi, Zachary C. Lipton, Arinbjorn Kolbeinsson, 
        Aran Khanna, Tommaso Furlanello, Anima Anandkumar, JMLRs, 2020. 

    .. [2] Tensor Regression Networks with various Low-Rank Tensor Approximations
        Xingwei Cao, Guillaume Rabusseau, 2018

    """
    def __init__(self, input_shape, output_shape, rank, bias=False, verbose=0, **kwargs):
        super().__init__(input_shape, output_shape, bias=bias, verbose=verbose, **kwargs)

        self.rank = validate_cp_rank(self.weight_shape, rank=rank)           

        self.weights = nn.Parameter(torch.Tensor(self.rank))
        self.factors = nn.ParameterList(nn.Parameter(torch.Tensor(s, self.rank)) for s in self.weight_shape)

        self.init_from_random(decompose_full_weight=False)

    def forward(self, x):
        weights, factors = self._process_decomposition()

        regression_weights = tl.cp_to_tensor((weights, factors))
        if self.bias is None:
            return tenalg.inner(x, regression_weights, n_modes=tl.ndim(x)-1)
        else:
            return tenalg.inner(x, regression_weights, n_modes=tl.ndim(x)-1) + self.bias

    def init_from_random(self, decompose_full_weight=True):
        if decompose_full_weight:
            full_weight = torch.normal(0.0, 0.02, size=self.weight_shape)
            self.init_from_tensor(full_weight)
        else:
            init.cp_init(self.weights, self.factors)
        if self.bias is not None:
            self.bias.data.zero_()

    def init_from_decomposition(self, weights, factors, bias=None):
        with torch.no_grad():
            for i, f in enumerate(factors):
                self.factors[i].data = f
            self.weights.data = weights
            if self.bias is not None and bias is not None:
                self.bias.data = bias.view(self.output_shape)
        
    def init_from_tensor(self, weight, bias=None,
                        decomposition_kwargs=dict(n_iter_max=10, init='random')):
        weights, factors = parafac(weight.data, rank=self.rank, verbose=self.verbose, **decomposition_kwargs)
        self.init_from_decomposition(weights, factors, bias)

    def init_from_linear(self, weight, bias=None):
        """Initialise the TRL from the weights of a fully connected layer
        """
        with torch.no_grad():
            weight = torch.t(weight).contiguous().view(self.weight_shape)

            if self.verbose:
                print(f'Initializing CPTRL from linear weight of shape {weight.shape} with rank-{self.rank} CP decomposition.')

            self.init_from_tensor(weight, bias)

        if self.verbose:
            print('TRL successfully initialized.')

    def full_weight(self):
        """Return the reconstructed weights from the low_rank

        Returns
        -------
        tensor :
            weights recoonstructed from the low-rank ones learnt
        """
        return tl.cp_to_tensor((self.weights, self.factors))

    def get_decomposition(self):
        return (self.weights, self.factors)

class TensorTrainTRL(BaseTRL):
    """Tensor Regression Layer with Tensor-Train weights [1]_, [2]_
        
    Parameters
    -----------
    input_shape : int iterable
        shape of the input, excluding batch size
    output_shape : int iterable
        shape of the output, excluding batch size
    verbose : int, default is 0
        level of verbosity
    rank : int 
        rank of the Tensor-Train / tt weights
    verbose : int, default is 1
        level of verbosity, if 0, no information will be printed

    See Also
    --------
    CPTRL
    TuckerTRL

    References
    ----------
    .. [1] Tensor Regression Networks, Jean Kossaifi, Zachary C. Lipton, Arinbjorn Kolbeinsson, 
        Aran Khanna, Tommaso Furlanello, Anima Anandkumar, JMLR 2020. 

    .. [2] Tensor Regression Networks with various Low-Rank Tensor Approximations
        Xingwei Cao, Guillaume Rabusseau, 2018
    """

    def __init__(self, input_shape, output_shape, rank, bias=False, verbose=0, **kwargs):
        super().__init__(input_shape, output_shape, bias=bias, verbose=verbose, **kwargs)

        self.rank = validate_tt_rank(self.weight_shape, rank=rank)           

        self.factors = nn.ParameterList()
        for i, s in enumerate(self.weight_shape):
            self.factors.append(nn.Parameter(torch.Tensor(self.rank[i], s, self.rank[i+1])))

        # Things like setting the tt_shape above are the init is not in the base class
        self.init_from_random(decompose_full_weight=False)

    def forward(self, x):
        factors = self._process_decomposition()

        regression_weights = tl.tt_to_tensor(factors)
        
        if self.bias is None:
            return tenalg.inner(x, regression_weights, n_modes=tl.ndim(x)-1)
        else:
            return tenalg.inner(x, regression_weights, n_modes=tl.ndim(x)-1) + self.bias
    
    def init_from_random(self, decompose_full_weight=True):
        if decompose_full_weight:
            full_weight = torch.normal(0.0, 0.02, size=self.weight_shape)
            self.init_from_tensor(full_weight)
        else:
            init.tt_init(self.factors)
        if self.bias is not None:
            self.bias.data.zero_()

    def init_from_decomposition(self, factors, bias=None):
        for i, factor in enumerate(factors):
                self.factors[i].data = factor
        
        if self.bias is not None and bias is not None:
            self.bias.data = bias.view(self.output_shape)

    def init_from_tensor(self, weight, bias=None, decomposition_kwargs=dict()):
        factors = tensor_train(weight.data, rank=self.rank, verbose=self.verbose, **decomposition_kwargs)

        self.init_from_decomposition(factors, bias=bias)

    def init_from_linear(self, weight, bias=None):
        """Initialise the TRL from the weights of a fully connected layer
        """
        weight = torch.t(weight).contiguous().view(self.weight_shape)
        
        self.init_from_tensor(weight, bias)

        if self.verbose:
            print(f'Initializing TensorTrainTRL from linear weight of shape {weight.shape} with rank-{self.rank} TT decomposition.')

        if self.verbose:
            print('TRL successfully initialized.')

    def full_weight(self):
        """Return the reconstructed weights from the low_rank

        Returns
        -------
        tensor :
            weights recoonstructed from the low-rank ones learnt
        """
        return tl.tt_to_tensor(self.factors)

    def get_decomposition(self):
        return self.factors