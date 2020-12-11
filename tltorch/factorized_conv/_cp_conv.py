"""
Higher Order Convolution with CP decompositon
"""

# Author: Jean Kossaifi
# License: BSD 3 clause

from ._base_conv import Conv1D, BaseFactorizedConv
from .. import init
from tensorly import validate_cp_rank
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math

import tensorly as tl
from tensorly.decomposition import parafac
from tensorly import random
from tensorly import tenalg
tl.set_backend('pytorch')


class CPConv(BaseFactorizedConv):
    """Create a Factorized CP convolution of arbitrary order.

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
    implementation = {'factorized', 'reconstructed', 'mobilenet'}
        Strategy to use for the forward pass. Options are:

        * factorized : the CP conv is expressed as a series of 1D convolutions
        * reconstructed : full kernel is reconstructed from the decomposition. 
            the reconstruction is used to perform a regular forward pass
        * mobilenet : the equivalent formulation of CP as a MobileNet block is used
        
    stride : int, default is 1
    padding : int, default is 0
    dilation : int, default is 0

    Attributes
    ----------
    
    kernel_shape : int tuple
        shape of the kernel weight parametrizing the full convolution
    rank : int
        rank of the CP decomposition

    References
    ----------

    .. [1] Vadim Lebedev, Yaroslav Ganin, Maksim Rakhuba, Ivan V.Oseledets, and Victor S. Lempitsky.
        Speeding-up convolu-tional neural networks using fine-tuned cp-decomposition. InICLR, 2015.

    .. [2] Jean Kossaifi, Antoine Toisoul, Adrian Bulat, Yannis Panagakis, Timothy M. Hospedales, Maja Pantic; 
        Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2020, pp. 6060-6069 

    See Also
    --------
    TuckerConv
    TTConvs
    """
    def __init__(self, in_channels, out_channels, kernel_size, rank, order=None,
                 implementation='reconstructed', stride=1, padding=0, dilation=1, bias=False):
        super().__init__(in_channels, out_channels, kernel_size, rank, order=order, padding=padding, stride=stride, bias=bias)
        self.rank = validate_cp_rank(self.kernel_shape, rank=self.rank)

        self.weights = nn.Parameter(torch.Tensor(self.rank))
        self.factors = nn.ParameterList([nn.Parameter(torch.Tensor(s, self.rank)) for s in self.kernel_shape])

        self.init_from_random(decompose_full_weight=False)

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
            init.cp_init(self.weights, self.factors)

    def init_from_decomposition(self, cp_tensor, bias=None):
        """Transpose the factors from a CP Tensor to the factorized version
    
        Parameters
        ----------
        factors : cp_tensor
        """
        shape, rank = tl.cp_tensor._validate_cp_tensor(cp_tensor)
        if shape != self.kernel_shape:
            raise ValueError(f'Expected a shape of {self.kernel_shape} but got {shape}.')
        if rank != self.rank:
            raise ValueError(f'Expected a cp_tensor of rank {self.rank} but got {rank}.')

        weights, factors = cp_tensor
    
        with torch.no_grad():
            for i, f in enumerate(factors):
                self.factors[i].data = f
            self.weights.data = weights
            if self.bias is not None and bias is not None:
                self.bias.data = bias.data

    def init_from_tensor(self, kernel_tensor, bias=None, decomposition_kwargs=dict()):
        """Initialize the factorized convolutional layer from a full tensor
        """
        with torch.no_grad():
            cp_tensor = parafac(kernel_tensor, rank=self.rank, **decomposition_kwargs)
            self.init_from_decomposition(cp_tensor, bias=bias)

    def get_decomposition(self, return_bias=False):
        """Returns a CP Tensor parametrizing the convolution
        
        Parameters
        ----------
        return_bias : bool, default is False
            if True also return the bias

        Returns
        -------
        weights, factors, bias
        """
        if return_bias:
            if self.bias is not None:
                bias = nn.Parameter(self.bias.data)
            else:
                bias = None
            return (self.weights, self.factors), bias
        else:
            return (self.weights, self.factors)

    def transduct(self, kernel_size, mode=0, padding=0, stride=1):
        """Transduction of the factorized convolution to add a new dimension

        Parameters
        ----------
        kernel_size : int
            size of the additional dimension
        mode : where to insert the new dimension, after the channels, default is 0
            by default, insert the new dimensions before the existing ones
            (e.g. add time before height and width)
        padding : int, default is 0
        stride : int: default is 1

        Returns
        -------
        self
        """
        (weights, factors), bias = self.get_decomposition(return_bias=True)
        self.order += 1
        self.padding = self.padding[:mode] + (padding,) + self.padding[mode:]
        self.stride = self.stride[:mode] + (stride,) + self.stride[mode:]
        self.kernel_size = self.kernel_size[:mode] + (kernel_size,) + self.kernel_size[mode:]

        self.kernel_shape = self.kernel_shape[:mode+2] + (kernel_size,) + self.kernel_shape[mode+2:]

        factors = [f for f in factors]
        # +2 corresponding to input and output channels
        #new_factor = torch.ones(kernel_size, self.rank)
        new_factor = torch.zeros(kernel_size, self.rank)
        new_factor[kernel_size//2, :] = 1

        factors.insert(mode+2, nn.Parameter(new_factor.to(self.factors[0].device)))
        self.init_from_decomposition((weights, factors), bias)

        return self
    

class CPConvFactorized(CPConv):
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
        rank of the CP decomposition
    """
    def __init__(self, in_channels, out_channels, kernel_size, rank, order=None, 
                 stride=1, padding=0, dilation=1, bias=False):
        super().__init__(in_channels, out_channels, kernel_size, rank, order=order, padding=padding, stride=stride, bias=bias)
        self.n_conv = len(self.kernel_shape)
    
    def forward(self, x):
        """Perform a factorized CP convolution

        Parameters
        ----------
        x : torch.tensor
            tensor of shape (batch_size, C, I_2, I_3, ..., I_N)

        Returns
        -------
        NDConv(x) with an CP kernel
        """
        weights, factors = self._process_decomposition()
        _, rank = tl.cp_tensor._validate_cp_tensor((weights, factors))

        batch_size = x.shape[0]
        # rank = self.rank

        # Change the number of channels to the rank
        x_shape = list(x.shape)
        x = x.reshape((batch_size, x_shape[1], -1)).contiguous()

        # First conv == tensor contraction
        # from (in_channels, rank) to (rank == out_channels, in_channels, 1)
        x = F.conv1d(x, tl.transpose(factors[1]).unsqueeze(2))

        x_shape[1] = rank
        x = x.reshape(x_shape)

        # convolve over non-channels
        for i in range(self.order):
            # From (kernel_size, rank) to (rank, 1, kernel_size)
            kernel = tl.transpose(factors[i+2]).unsqueeze(1)             
            x = Conv1D(x.contiguous(), kernel, i+2, stride=self.stride[i], padding=self.padding[i], groups=rank)

        # Revert back number of channels from rank to output_channels
        x_shape = list(x.shape)
        x = x.reshape((batch_size, x_shape[1], -1))                
        # Last conv == tensor contraction
        # From (out_channels, rank) to (out_channels, in_channels == rank, 1)
        x = F.conv1d(x*weights.unsqueeze(1).unsqueeze(0), factors[0].unsqueeze(2))

        if self.bias is not None:
            x += self.bias.unsqueeze(0).unsqueeze(2)

        x_shape[1] = self.out_channels
        x = x.reshape(x_shape)

        return x



class CPConvMobileNet(CPConv):
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
        rank of the CP decomposition
    """
    def __init__(self, in_channels, out_channels, kernel_size, rank, order=None, 
                 stride=1, padding=0, dilation=1, bias=False):
        super().__init__(in_channels, out_channels, kernel_size, rank, order=order, padding=padding, stride=stride, bias=bias)
        self.n_conv = len(self.kernel_shape)

    def forward(self, x):
        """Perform a factorized CP convolution

        Parameters
        ----------
        x : torch.tensor
            tensor of shape (batch_size, C, I_2, I_3, ..., I_N)

        Returns
        -------
        NDConv(x) with an CP kernel
        """
        weights, factors = self._process_decomposition()
        _, rank = tl.cp_tensor._validate_cp_tensor((weights, factors))

        batch_size = x.shape[0]
        # rank = self.rank

        # Change the number of channels to the rank
        x_shape = list(x.shape)
        x = x.reshape((batch_size, x_shape[1], -1)).contiguous()

        # First conv == tensor contraction
        # from (in_channels, rank) to (rank == out_channels, in_channels, 1)
        x = F.conv1d(x, tl.transpose(factors[1]).unsqueeze(2))

        x_shape[1] = rank
        x = x.reshape(x_shape)

        # convolve over merged actual dimensions
        # Spatial convs
        # From (kernel_size, rank) to (out_rank, 1, kernel_size)
        if self.order == 1:
            weight = tl.transpose(factors[2]).unsqueeze(1)
            x = F.conv1d(x.contiguous(), weight, stride=self.stride, padding=self.padding, dilation=self.dilation, groups=rank)
        elif self.order == 2:
            weight = tenalg.batched_tensor_dot(tl.transpose(factors[2]), tl.transpose(factors[3])).unsqueeze(1)
            x = F.conv2d(x.contiguous(), weight, stride=self.stride, padding=self.padding, dilation=self.dilation, groups=rank)
        elif self.order == 3:
            weight = tenalg.batched_tensor_dot(tl.transpose(factors[2]), 
                        tenalg.batched_tensor_dot(tl.transpose(factors[3]), tl.transpose(factors[4]))).unsqueeze(1)
            x = F.conv3d(x.contiguous(), weight, stride=self.stride, padding=self.padding, dilation=self.dilation, groups=rank)

        # Revert back number of channels from rank to output_channels
        x_shape = list(x.shape)
        x = x.reshape((batch_size, x_shape[1], -1))

        # Last conv == tensor contraction
        # From (out_channels, rank) to (out_channels, in_channels == rank, 1)
        x = F.conv1d(x*weights.unsqueeze(1).unsqueeze(0), factors[0].unsqueeze(2))

        if self.bias is not None:
            x += self.bias.unsqueeze(0).unsqueeze(2)

        x_shape[1] = self.out_channels
        x = x.reshape(x_shape)

        return x

class CPConvReconstructed(CPConv):
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
        rank of the CP decomposition
    """
    def __init__(self, in_channels, out_channels, kernel_size, rank, order=None, 
                 stride=1, padding=0, dilation=1, bias=False):
        super().__init__(in_channels, out_channels, kernel_size, rank, order=order, padding=padding, stride=stride, bias=bias)
        self.n_conv = len(self.kernel_shape)
        if self.order == 1: 
            self.conv = F.conv1d
        elif self.order == 2:
            self.conv = F.conv2d
        elif self.order == 3:
            self.conv = F.conv3d
        else:
            raise ValueError(f'{self.__class__.__name__} currently implemented only for 1D to 3D convs, but got {order}')

    def forward(self, x):
        """Perform a convolution using the reconstructed full weightss

        Parameters
        ----------
        x : torch.tensor
            tensor of shape (batch_size, C, I_2, I_3, ..., I_N)

        Returns
        -------
        NDConv(x) with a CP kernel
        """
        rec = tl.cp_to_tensor(self._process_decomposition())
        return self.conv(x, rec, bias=self.bias, stride=self.stride, padding=self.padding, dilation=self.dilation)

    def transduct(self, *args, **kwargs):
        super().transduct(*args, **kwargs)
        if self.order == 1: 
            self.conv = F.conv1d
        elif self.order == 2:
            self.conv = F.conv2d
        elif self.order == 3:
            self.conv = F.conv3d
