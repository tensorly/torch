"""
Higher Order Convolution with CP decompositon
"""

# Author: Jean Kossaifi
# License: BSD 3 clause

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import warnings

import tensorly as tl
tl.set_backend('pytorch')
from ..base import TensorModule

def Conv1D(x, kernel, mode, stride=1, padding=0, groups=1, verbose=False):
    """General 1D convolution along the mode-th dimension

    Parameters
    ----------
    x : batch-dize, in_channels, K1, ..., KN
    kernel : out_channels, in_channels/groups, K{mode}
    mode : int
        weight along which to perform the decomposition
    stride : int
    padding : int
    groups : 1
        typically would be equal to thhe number of input-channels
        at least for CP convolutions

    Returns
    -------
    x convolved with the given kernel, along dimension `mode`
    """
    if verbose:
        print(f'Convolving {x.shape} with {kernel.shape} along mode {mode}, '
              f'stride={stride}, padding={padding}, groups={groups}')

    in_channels = tl.shape(x)[1]
    n_dim = tl.ndim(x)
    permutation = list(range(n_dim))
    spatial_dim = permutation.pop(mode)
    channels_dim = permutation.pop(1)
    permutation += [channels_dim, spatial_dim]
    x = tl.transpose(x, permutation)
    x_shape = list(x.shape)
    x = tl.reshape(x, (-1, in_channels, x_shape[-1]))
    x = F.conv1d(x.contiguous(), kernel, stride=stride, padding=padding, groups=groups)
    x_shape[-2:] = x.shape[-2:]
    x = tl.reshape(x, x_shape)
    permutation = list(range(n_dim))[:-2]
    permutation.insert(1, n_dim - 2)
    permutation.insert(mode, n_dim - 1)
    x = tl.transpose(x, permutation)
    
    return x


class MetaFactorizedConv(type):
    """Meta class for factorized convolutions
    
        Calls __new__ with `implementation` in kwargs
        Calls __init__ after removing `implementation` from kwargs

        Errors if the created instance doesn't have an attribute 'implementation'

    Notes
    -----
    implementations are in the meta class to be both classattributes but also properties
    """
    def __call__(cls, *args, **kwargs):
        instance = cls.__new__(cls, *args, **kwargs)
        try:
            kwargs.pop('implementation')
        except KeyError:
            assert hasattr(instance, 'implementation'), f'No implementation was set in {instance}'

        instance.__init__(*args, **kwargs)

        return instance

    @property
    def implementations(cls):
        return list(cls._implementations.keys())

    @property
    def default_implementation(cls):
        return cls._default_implementation
    
    @default_implementation.setter
    def default_implementation(cls, value):
        if value not in cls._implementations.keys():
            raise ValueError(f'implementation should be one of {cls.implementations}, but got {value}.')
        cls._default_implementation = value


class BaseFactorizedConv(TensorModule, metaclass=MetaFactorizedConv):
    """Create a convolution of arbitrary order

    Parameters
    ----------
    in_channels : int
    out_channels : int
    kernel_size : int or int list
        if int, order MUST be specified
        if int list, then the conv will use order = len(kernel_size)
    rank : int
        rank of the factorized kernel
    order : int, optional if kernel_size is a list
        see kernel_size
    stride : int, default is 1
    padding : int, default is 0
    dilation : int, default is 0
    """

    _version = 1
    # init implementations to None, not dict, so each subclass gets a different one
    # Otherwise the dict's values would be shared among all subclasses
    _implementations = None
    _default_implementation = None

    def __new__(cls, *args, **kwargs):
        """Customize the creation of a factorized convolution

            Takes in a string `implementation` that specifies with subclass to use

        Returns
        -------
        BaseFactorizedConv._implementations[implementation]
            subclass implementing the specified factorized conv
        """
        try:
            implementation = kwargs.pop('implementation')
        except KeyError:
            warnings.warn(f'No value provided for implementation, using default={cls._default_implementation}.')
            implementation = cls._default_implementation
        instance = super().__new__(cls._implementations[implementation])
        instance.implementation = implementation
        return instance

    @classmethod
    def register_implementation(cls, name, value, set_default=False):
        """Register a new implementation (subclass) of the Factorized convolution

        Parameters
        ----------
        name : str
            name of the implementation
        value : class 
            subclass of BaseFactorizedConvolution
        set_default : bool, default is True
            whether to set the implementation as the default one
            will be set to True if no implementation was already registered

        Notes
        -----
        No init is done here, in the base class, in case more variables need to be set first, 
        see TTConv
        """
        if cls._implementations is None:
            cls._implementations = {name:value}
        else:
            cls._implementations[name] = value
        if set_default or cls._default_implementation is None:
            cls._default_implementation = name
    
    def __init__(self, in_channels, out_channels, kernel_size, rank, order=None, 
                 stride=1, padding=0, dilation=1, bias=False):
        super().__init__()

        if isinstance(kernel_size, int):
            if order is None:
                raise ValueError(f'If int given for kernel_size, order (dimension of the convolution) should also be provided.')
            if not isinstance(order, int) or order <= 0:
                raise ValueError(f'order should be the (positive integer) order of the convolution'
                                 f'but got order={order} of type {type(order)}.')
            else:
                kernel_size = (kernel_size, )*order
        else:
            kernel_size = tuple(kernel_size)
            order = len(kernel_size)

        kernel_shape = (out_channels, in_channels) + kernel_size

        if isinstance(padding, int):
            padding = (padding, )*order

        if isinstance(stride, int):
            stride = (stride, )*order

        self.kernel_size = kernel_size
        self.kernel_shape = kernel_shape
        self.rank = rank
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.padding = padding
        self.stride = stride
        self.dilation = dilation
        self.order = order
        self.decomposition_callback = []
        
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

    @classmethod 
    def from_conv(cls, conv_layer, rank='same', implementation='reconstructed', decompose_weights=True, modes_fixed_rank=None, decomposition_kwargs=dict()):
        """Create a Factorized convolution from a regular convolutional layer
        
        Parameters
        ----------
        conv_layer : torch.nn.ConvND
        rank : rank of the decomposition, default is 'same'
        implementation : str, default is 'reconstructed'
        decomposed_weights : bool, default is True
            if True, the convolutional kernel is decomposed to initialize the factorized convolution
            otherwise, the factorized convolution's parameters are initialized randomly
        decomposition_kwargs : dict 
            parameters passed directly on to the decompoosition function if `decomposed_weights` is True
        
        Returns
        -------
        New instance of the factorized convolution with equivalent weightss

        Todo
        ----
        Check that the decomposition of the given convolution and cls is the same.
        """
        kernel_tensor = conv_layer.weight
        padding = conv_layer.padding
        kernel_shape = kernel_tensor.shape
        out_channels, in_channels, *kernel_size = kernel_shape
        stride = conv_layer.stride[0]
        bias = conv_layer.bias is not None

        if modes_fixed_rank is not None:
            instance = cls(in_channels, out_channels, kernel_size, implementation=implementation, rank=rank, 
                padding=padding, stride=stride,  bias=bias, modes_fixed_rank=modes_fixed_rank)
        else:
            instance = cls(in_channels, out_channels, kernel_size, implementation=implementation, rank=rank, 
                        padding=padding, stride=stride,  bias=bias)

        if decompose_weights:
            if bias:
                bias = conv_layer.bias.data
            else:
                bias = None

            instance.init_from_tensor(kernel_tensor, bias=bias, **decomposition_kwargs)

        return instance
    
    @classmethod
    def from_factorized_conv(cls, conv_layer, implementation='reconstructed'):
        """Create a Factorized convolution from another factorized convolution
        
        Returns
        -------
        New instance of the factorized convolution with equivalent weights

        Todo
        ----
        Check that the decomposition of the given convolution and cls is the same.
        If not, recompose full weight and re-decompose
        """
        if not isinstance(conv_layer, cls.__class__):
            raise ValueError(f'Expected {cls.__class__.__name__} as input but got {conv_layer.__class__.__name__}.')
        instance = cls(conv_layer.in_channels, conv_layer.out_channels, conv_layer.kernel_size, 
                       implementation=implementation, rank=conv_layer.rank, 
                       padding=conv_layer.padding, stride=conv_layer.stride, bias=conv_layer.bias is not None)
        decomposition, bias = conv_layer.get_decomposition(return_bias=True)
        instance.init_from_decomposition(decomposition, bias=bias)
        return instance

    def init_from_tensor(self, kernel_tensor, bias, rank='same', decomposition_kwargs={}):
        raise NotImplementedError()

    def init_from_decomposition(self, decomposed_tensor, bias=None):
        raise NotImplementedError()

    def init_from_random(self, decompose_full_weight=True):
        if decompose_full_weight:
            full_weight = torch.normal(0.0, 0.02, size=self.weight_shape)
            self.init_from_tensor(full_weight)
        else:
            raise NotImplementedError()

    def __repr__(self):
        msg = f'{self.__class__.__name__},'
        msg += f' order={self.order}, rank={self.rank}, kernel_size={self.kernel_size},'
        msg += f' padding={self.padding}, stride={self.stride}, bias={self.bias is not None}.'
        return msg