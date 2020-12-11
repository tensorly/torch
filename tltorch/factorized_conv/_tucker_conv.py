"""
Higher Order Convolution with Tucker decompositon
"""

# Author: Jean Kossaifi
# License: BSD 3 clause

from ._base_conv import Conv1D, BaseFactorizedConv
from .. import init


import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math

import tensorly as tl; tl.set_backend('pytorch')
from tensorly import validate_tucker_rank
from tensorly.decomposition import tucker
from tensorly import random
from tensorly import tenalg


class TuckerConv(BaseFactorizedConv):
    """Create a convolution of arbitrary order with a Tucker kernel

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
    implementation = {'factorized', 'reconstructed'}
        strategy to use for the forward pass: 
        - factorized : the Tucker conv is expressed as a series of smaller convolutions
        - reconstructed : full kernel is reconstructed from the decomposition. 
          the reconstruction is used to perform a regular forward pass
    stride : int, default is 1
    padding : int, default is 0
    dilation : int, default is 0
    modes_fixed_rank : None, 'spatial' or int list, default is None
        if 'spatial', the rank along the spatial dimensions is kept the same as the original dimension
        if int list, the rank is kept fixed (same as input) along the specified modes
        otherwise (if None), all the ranks are determing

        *used only if rank is 'same' or a float.*
    
    Attributes
    ----------
    kernel_shape : int tuple
        shape of the kernel weight parametrizing the full convolution
    rank : int
        rank of the Tucker decomposition

    References
    ----------

    .. [1] Yong-Deok  Kim,  Eunhyeok  Park,  Sungjoo  Yoo,  TaelimChoi,  Lu  Yang,  and  Dongjun  Shin.   
        Compression of deep convolutional neural networks for fast and low power mobile applications. In ICLR, 2016.

    .. [2] Jean Kossaifi, Antoine Toisoul, Adrian Bulat, Yannis Panagakis, Timothy M. Hospedales, Maja Pantic; 
        Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2020, pp. 6060-6069 

    See Also
    --------
    CPConv
    TTConvs
    """
    def __init__(self, in_channels, out_channels, kernel_size, rank, modes_fixed_rank=None, order=None,
                 implementation='reconstructed', stride=1, padding=0, dilation=1, bias=False):
        super().__init__(in_channels, out_channels, kernel_size, rank, order=order, padding=padding, stride=stride, bias=bias)
        self.rank = validate_tucker_rank(self.kernel_shape, rank=self.rank, fixed_modes=modes_fixed_rank)

        if modes_fixed_rank is None:
            self.modes_fixed_rank = None
        elif modes_fixed_rank== 'spatial':
            self.modes_fixed_rank = list(range(2, 2+self.order))
        else:
            self.modes_fixed_rank = modes_fixed_rank

        self.core = nn.Parameter(torch.Tensor(self.rank))
        self.factors = nn.ParameterList(nn.Parameter(torch.Tensor(s, r))\
                                        for (s, r) in zip(self.kernel_shape, self.rank))

        self.init_from_random(decompose_full_weight=False)

    def init_from_tensor(self, kernel_tensor, bias=None, decomposition_kwargs=dict()):
        """Initializes the factorized convolution by decomposing a full tensor.
    
        Parameters
        ----------
        kernel_tensor : full convolutional kernel to decompose
        bias : optional, default is None
        decomposition_kwargs : dict
            dictionary of parameters passed on to the decomposition
        """
        with torch.no_grad():
            tucker_tensor = tucker(kernel_tensor, rank=self.rank,  **decomposition_kwargs)
            self.init_from_decomposition(tucker_tensor, bias=bias)
    
    def init_from_random(self, decompose_full_weight=True):
        """Initialize the factorized convolution's parameter randomly
        
        Parameters
        ----------
        decompose_full_weight : bool 
            If True, a full weight is randomly created and decomposed to intialize the parameters (slower)
            Otherwise, the parameters are initialized directly (faster) so the reconstruction has a set variance. 
        """
        if self.bias is not None:
            self.bias.data.zero_()

        if decompose_full_weight:
            full_weight = torch.normal(0.0, 0.02, size=self.kernel_shape)
            self.init_from_tensor(full_weight)
        else:
            init.tucker_init(self.core, self.factors)

    def init_from_decomposition(self, tucker_tensor, bias=None):
        """Initialize the factorized convolution with a decomposed tensor
        
        Parameters
        ----------
        factors : Tucker tensor 
            (core, factors) of the Tucker tensor
        """
        shape, rank = tl.tucker_tensor._validate_tucker_tensor(tucker_tensor)
        self.rank = rank
        if shape != self.kernel_shape:
            raise ValueError(f'Expected a shape of {self.kernel_shape} but got {shape}.')
        core, factors = tucker_tensor
        
        with torch.no_grad():
            for i, f in enumerate(factors):
                self.factors[i].data = f
            self.core.data = core

        if self.bias is not None and bias is not None:
            self.bias.data = bias

    def get_decomposition(self, return_bias=False):
        """Returns the Tucker Tensor parametrizing the convolution

        Parameters
        ----------
        return_bias : bool, default is False
            if True also return the bias

        Returns
        -------
        (core, factors), bias if return_bias:
        (core, factors) otherwise
        """
        if return_bias:
            return (self.core, self.factors), self.bias
        else:
            return (self.core, self.factors)


class TuckerConvFactorized(TuckerConv):
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
    
    Attributes
    ----------
    kernel_shape : int tuple
        shape of the kernel weight parametrizing the full convolution
    rank : int
        rank of the Tucker decomposition
    """
    def __init__(self, in_channels, out_channels, kernel_size, rank, order=None, modes_fixed_rank=None,
                 stride=1, padding=0, dilation=1, bias=False):
        super().__init__(in_channels, out_channels, kernel_size, rank, order=order, padding=padding, stride=stride, bias=bias, modes_fixed_rank=modes_fixed_rank)
        self.n_conv = len(self.kernel_shape)
        if self.order == 1: 
            self.conv = F.conv1d
        elif self.order == 2:
            self.conv = F.conv2d
        elif self.order == 3:
            self.conv = F.conv3d
        else:
            raise ValueError(f'{self.__class__.__name__} currently implemented only for 1D to 3D convs, but got {self.order}')
        
    def forward(self, x):
        """Perform a factorized Tucker convolution

        Parameters
        ----------
        x : torch.tensor
            tensor of shape (batch_size, C, I_2, I_3, ..., I_N)

        Returns
        -------
        NDConv(x) with an Tucker kernel
        """
        core, factors = self._process_decomposition()
        # Extract the rank from the actual decomposition in case it was changed by, e.g. dropout
        _, rank = tl.tucker_tensor._validate_tucker_tensor((core, factors))

        batch_size = x.shape[0]
        # rank = self.rank
        n_dim = tl.ndim(x)

        # Change the number of channels to the rank
        x_shape = list(x.shape)
        x = x.reshape((batch_size, x_shape[1], -1)).contiguous()

        # This can be done with a tensor contraction
        # First conv == tensor contraction
        # from (in_channels, rank) to (rank == out_channels, in_channels, 1)
        x = F.conv1d(x, tl.transpose(factors[1]).unsqueeze(2))

        x_shape[1] = rank[1]
        x = x.reshape(x_shape)

        modes = list(range(2, n_dim+1))
        weight = tl.tenalg.multi_mode_dot(core, factors[2:], modes=modes)
        x = self.conv(x, weight, bias=None, stride=self.stride, padding=self.padding)

        # Revert back number of channels from rank to output_channels
        x_shape = list(x.shape)
        x = x.reshape((batch_size, x_shape[1], -1))
        # Last conv == tensor contraction
        # From (out_channels, rank) to (out_channels, in_channels == rank, 1)
        x = F.conv1d(x, factors[0].unsqueeze(2))

        if self.bias is not None:
            x += self.bias.unsqueeze(0).unsqueeze(2)

        x_shape[1] = self.out_channels
        x = x.reshape(x_shape)

        return x


class TuckerConvReconstructed(TuckerConv):
    """Create a Tucker convolution of arbitrary order

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
    
    Attributes
    ----------
    kernel_shape : int tuple
        shape of the kernel weight parametrizing the full convolution
    rank : int
        rank of the Tuckers decomposition
    """
    def __init__(self, in_channels, out_channels, kernel_size, rank, order=None, modes_fixed_rank=None,
                 stride=1, padding=0, dilation=1, bias=False):
        super().__init__(in_channels, out_channels, kernel_size, rank, order=order, padding=padding, stride=stride, bias=bias, modes_fixed_rank=modes_fixed_rank)
        self.n_conv = len(self.kernel_shape)
        if self.order == 1: 
            self.conv = F.conv1d
        elif self.order == 2:
            self.conv = F.conv2d
        elif self.order == 3:
            self.conv = F.conv3d
        else:
            raise ValueError(f'{self.__class__.__name__} currently implemented only for 1D to 3D convs, but got {self.order}')
    
    def forward(self, x):
        """Perform a convolution using the reconstructed full weightss

        Parameters
        ----------
        x : torch.tensor
            tensor of shape (batch_size, C, I_2, I_3, ..., I_N)

        Returns
        -------
        NDConv(x) with a Tucker kernel
        """
        rec = tl.tucker_to_tensor(self._process_decomposition())
        return self.conv(x, rec, bias=self.bias, stride=self.stride, padding=self.padding, dilation=self.dilation)


