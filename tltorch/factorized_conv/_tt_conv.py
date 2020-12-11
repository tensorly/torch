"""
Higher Order Convolution with Tensor-Train decompositon
"""

# Author: Jean Kossaifi
# License: BSD 3 clause

from ._base_conv import Conv1D, BaseFactorizedConv
from .. import init

from tensorly import validate_tt_rank
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math

import tensorly as tl
from tensorly.decomposition import tensor_train
from tensorly import random
from tensorly import tenalg
tl.set_backend('pytorch')


class TTConv(BaseFactorizedConv):
    """Create a convolution of arbitrary order with a Tucker kernel.

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
        strategy to use for the forward pass
        - factorized : the TT conv is expressed as a series of 1D convolutions
        - reconstructed : full kernel is reconstructed from the decomposition. 
        the reconstruction is used to perform a regular forward pass
    stride : int, default is 1
    padding : int, default is 0
    dilation : int, default is 0

    Attributes
    ----------
    kernel_shape : int tuple
        shape of the kernel weight parametrizing the full convolution
    rank : int
        rank of the TT decomposition

    See Also
    --------
    TuckerConv
    CPConv

    References
    ----------
    
    .. [2] Jean Kossaifi, Antoine Toisoul, Adrian Bulat, Yannis Panagakis, Timothy M. Hospedales, Maja Pantic; 
        Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2020, pp. 6060-6069 

    """
    def __init__(self, in_channels, out_channels, kernel_size, rank, order=None,
                 implementation=None, stride=1, padding=0, dilation=1, bias=False):
        super().__init__(in_channels, out_channels, kernel_size, rank, order=order, padding=padding, stride=stride, bias=bias)
        tt_shape = list(self.kernel_shape)
        self.n_conv = len(tt_shape)

        # For the TT case, the decomposition has a different shape than the kernel.
        out_channel = tt_shape.pop(0)
        tt_shape += [out_channel]
        self.tt_shape = tuple(tt_shape)
        self.rank = tl.tt_tensor.validate_tt_rank(self.tt_shape, rank)

        self.factors = nn.ParameterList()
        for i, s in enumerate(self.tt_shape):
            self.factors.append(nn.Parameter(torch.Tensor(self.rank[i], s, self.rank[i+1])))

        # Things like setting the tt_shape above are the init is not in the base class
        self.init_from_random(decompose_full_weight=False)


    def init_from_tensor(self, kernel_tensor, bias=None, decomposition_kwargs=dict()):
        """Initialize the factorized convolutional layer from a regular convolutional layer
        """        
        self.rank = validate_tt_rank(kernel_tensor.shape, rank=self.rank)

        with torch.no_grad():
            # Put output channels at the end
            kernel_tensor = tl.moveaxis(kernel_tensor, 0, -1)
            tt_tensor = tensor_train(kernel_tensor, rank=self.rank, **decomposition_kwargs)
            self.init_from_decomposition(tt_tensor, bias=bias)
    
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
            full_weight = torch.normal(0.0, 0.02, size=self.tt_shape)
            self.init_from_tensor(full_weight)
        else:
            init.tt_init(self.factors)

    def init_from_decomposition(self, tt_tensor, bias=None):
        """Transpose the factors from a full weight to the factorized version
        
        Parameters
        ----------
        factors : tt_tensor
        """
        shape, rank = tl.tt_tensor._validate_tt_tensor(tt_tensor)
        self.rank = rank
        if shape != self.tt_shape:
            raise ValueError(f'Expected a shape of {self.tt_shape} but got {shape}.')
        
        with torch.no_grad():
            for i, f in enumerate(tt_tensor):
                self.factors[i].data = f
        
            if self.bias is not None and bias is not None:
                self.bias.data = bias

    def get_decomposition(self, return_bias=False):
        """Transpose back factors from a factorized version
        
        Parameters
        ----------
        return_bias : bool, default is False
            if True also return the bias
        
        Returns
        -------
        factors, bias if return_bias:
        factors otherwise
        """
        if return_bias:
            return self.factors, self.bias
        else:
            return self.factors

    def full_weights(self):
        """Returns the reconstructed full convolutional kernel 
        """
        factors = self.get_decomposition(return_bias=False)
        kernel = tl.tt_to_tensor(factors)
        return tl.moveaxis(kernel, -1, 0)

    def transduct(self, kernel_size, mode=0, padding=0, stride=1, dilation=1):
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
        factors, bias = self.get_decomposition(return_bias=True)
        # Increase the order of the convolution
        self.order += 1
        self.n_conv += 1
        self.padding = self.padding[:mode] + (padding,) + self.padding[mode:]
        self.stride = self.stride[:mode] + (stride,) + self.stride[mode:]
        self.kernel_size = self.kernel_size[:mode] + (kernel_size,) + self.kernel_size[mode:]
        #self.dilation = self.dilation[:mode] + (dilation,) + self.dilation[mode:]
        
        self.kernel_shape = self.kernel_shape[:mode+2] + (kernel_size,) + self.kernel_shape[mode+2:]
        # tt_shape is (in_channels, spacial_dims, out_channels)
        self.tt_shape = self.tt_shape[:mode+1] + (kernel_size, ) + self.tt_shape[mode+1:]
        # rank[0] = rank[-1] = 1
        new_rank = self.rank[mode+1]
        self.rank = self.rank[:mode+1] + (new_rank, ) + self.rank[mode+2:]

        factors = [f for f in factors]
        # +1 -- account for input channels
        new_factor = torch.zeros(new_rank, kernel_size, new_rank)
        for i in range(kernel_size):
            new_factor[:, i, :] = torch.eye(new_rank)
        factors.insert(mode+1, nn.Parameter(new_factor.to(self.factors[0].device)))  
        
        self.init_from_decomposition(factors, bias)

        return self
        

class TTConvFactorized(TTConv):
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
        rank of the tt decomposition
    """
    def __init__(self, in_channels, out_channels, kernel_size, rank, order=None, 
                 stride=1, padding=0, dilation=1, bias=False):
        super().__init__(in_channels, out_channels, kernel_size, rank, order=order, padding=padding, stride=stride, bias=bias)
        # tt_shape = list(self.kernel_shape)
        # self.n_conv = len(tt_shape)

        # out_channel = tt_shape.pop(0)
        # tt_shape += [out_channel]
        # self.tt_shape = tuple(tt_shape)
    
    def forward(self, x):
        """Perform a factorized tt convolution

        Parameters
        ----------
        x : torch.tensor
            tensor of shape (batch_size, C, I_2, I_3, ..., I_N)

        Returns
        -------
        NDConv(x) with an tt kernel
        """
        factors = self._process_decomposition()
        _, rank = tl.tt_tensor._validate_tt_tensor(factors)


        batch_size = x.shape[0]
        # rank = self.rank

        # Change the number of channels to the rank
        x_shape = list(x.shape)
        x = x.reshape((batch_size, x_shape[1], -1)).contiguous()

        # First conv == tensor contraction
        # from (1, in_channels, rank) to (rank == out_channels, in_channels, 1)
        x = F.conv1d(x, tl.transpose(factors[0], [2, 1, 0]))

        x_shape[1] = rank[1]
        x = x.reshape(x_shape)

        # convolve over non-channels
        for i in range(self.n_conv-2):
            # From (in_rank, kernel_size, out_rank) to (out_rank, in_rank, kernel_size)
            kernel = tl.transpose(factors[i+1], [2, 0, 1])
            x = Conv1D(x.contiguous(), kernel, i+2, stride=self.stride[i], padding=self.padding[i])#, groups=self.rank[i+1])

        # Revert back number of channels from rank to output_channels
        x_shape = list(x.shape)
        x = x.reshape((batch_size, x_shape[1], -1))
        # Last conv == tensor contraction
        # From (rank, out_channels, 1) to (out_channels, in_channels == rank, 1)
        x = F.conv1d(x, tl.transpose(factors[-1], [1, 0, 2]))

        if self.bias is not None:
            x += self.bias.unsqueeze(0).unsqueeze(2)

        x_shape[1] = self.out_channels
        x = x.reshape(x_shape)

        return x


class TTConvReconstructed(TTConv):
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
        rank of the tt decomposition
    """
    def __init__(self, in_channels, out_channels, kernel_size, rank, order=None, 
                 stride=1, padding=0, dilation=1, bias=False):
        super().__init__(in_channels, out_channels, kernel_size, rank, order=order, padding=padding, stride=stride, bias=bias)
        # tt_shape = list(self.kernel_shape)
        # self.n_conv = len(tt_shape)

        # out_channel = tt_shape.pop(0)
        # tt_shape += [out_channel]
        # self.tt_shape = tuple(tt_shape)
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
        """Perform a factorized tt convolution

        Parameters
        ----------
        x : torch.tensor
            tensor of shape (batch_size, C, I_2, I_3, ..., I_N)

        Returns
        -------
        NDConv(x) with an tt kernel
        """
        rec = tl.moveaxis(tl.tt_to_tensor(self._process_decomposition()), -1, 0) 
        return self.conv(x, rec, bias=self.bias, stride=self.stride, padding=self.padding, dilation=self.dilation)

    def transduct(self, *args, **kwargs):
        super().transduct(*args, **kwargs)
        if self.order == 1: 
            self.conv = F.conv1d
        elif self.order == 2:
            self.conv = F.conv2d
        elif self.order == 3:
            self.conv = F.conv3d

