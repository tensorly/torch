import math
import numpy as np
from torch import nn
import torch

from ..functional import factorized_linear
from ..factorized_tensors import TensorizedTensor

# Author: Jean Kossaifi
# License: BSD 3 clause


class FactorizedLinear(nn.Module):
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
    def __init__(self, in_tensorized_features, out_tensorized_features, bias=True,
                 factorization='cp', rank='same', n_layers=1, device=None, dtype=None):
        super().__init__()
        if factorization == 'TTM' and n_layers != 1:
            raise ValueError(f'TTM factorization only support single factorized layers but got n_layers={n_layers}.')

        self.in_features = np.prod(in_tensorized_features)
        self.out_features = np.prod(out_tensorized_features)
        self.in_tensorized_features = in_tensorized_features
        self.out_tensorized_features = out_tensorized_features
        self.tensorized_shape = out_tensorized_features + in_tensorized_features
        self.weight_shape = (self.out_features, self.in_features)
        self.input_rank = rank

        if bias:
            if n_layers == 1:
                self.bias = nn.Parameter(torch.empty(self.out_features, device=device, dtype=dtype))
                self.has_bias = True
            else:
                self.bias = nn.Parameter(torch.empty((n_layers, self.out_features), device=device, dtype=dtype))
                self.has_bias = np.zeros(n_layers)
        else:
            self.register_parameter('bias', None)

        self.rank = rank
        self.n_layers = n_layers
        if n_layers > 1:
            tensor_shape = (n_layers, out_tensorized_features, in_tensorized_features)
        else:
            tensor_shape = (out_tensorized_features, in_tensorized_features)
        
        if isinstance(factorization, TensorizedTensor):
            self.weight = factorization.to(device).to(dtype)
        else:
            self.weight = TensorizedTensor.new(tensor_shape, rank=rank, factorization=factorization, device=device, dtype=dtype)
            self.reset_parameters()

        self.rank = self.weight.rank

    def reset_parameters(self):
        with torch.no_grad():
            self.weight.normal_(0, math.sqrt(5)/math.sqrt(self.in_features))
            if self.bias is not None:
                fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(self.weight)
                bound = 1 / math.sqrt(fan_in)
                torch.nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x, indices=0):
        if self.n_layers == 1:
            if indices == 0:
                return factorized_linear(x, self.weight(), bias=self.bias, in_features=self.in_features)

            else:
                raise ValueError(f'Only one convolution was parametrized (n_layers=1) but tried to access {indices}.')

        elif isinstance(self.n_layers, int):
            if not isinstance(indices, int):
                raise ValueError(f'Expected indices to be in int but got indices={indices}'
                                 f', but this conv was created with n_layers={self.n_layers}.')
            bias = self.bias[indices] if self.bias is not None else None
            return factorized_linear(x, self.weight(indices), bias=bias, in_features=self.in_features)

        elif len(indices) != len(self.n_layers):
            raise ValueError(f'Got indices={indices}, but this conv was created with n_layers={self.n_layers}.')

        else:

            bias = self.bias[indices] if self.bias is not None else None
            return factorized_linear(x, self.weight(indices), bias=bias, in_features=self.in_features)

    def get_linear(self, indices):
        if self.n_layers == 1:
            raise ValueError('A single linear is parametrized, directly use the main class.')

        return SubFactorizedLinear(self, indices)

    def __getitem__(self, indices):
        return self.get_linear(indices)

    @classmethod
    def from_linear(cls, linear, in_tensorized_features, out_tensorized_features, rank, bias=True,
                    factorization='CP', decomposition_kwargs=dict()):
        """Class method to create an instance from an existing linear layer
        
        Parameters
        ----------
        linear : torch.nn.Linear
            layer to tensorize
        tensorized_shape : tuple
            shape to tensorized the factorized_weight matrix to.
            Must verify np.prod(tensorized_shape) == np.prod(linear.factorized_weight.shape)
        rank :  {rank of the decomposition, 'same', float}
            if float, percentage of parameters of the original factorized_weights to use
            if 'same' use the same number of parameters
        bias : bool, default is True
        """
        out_features, in_features = linear.weight.shape
        assert(out_features == np.prod(out_tensorized_features))
        assert(in_features == np.prod(in_tensorized_features))

        instance = cls(in_tensorized_features, out_tensorized_features, bias=bias,
                       factorization=factorization, rank=rank, n_layers=1,
                       device=linear.weight.device, dtype=linear.weight.dtype)

        instance.weight.init_from_matrix(linear.weight.data, **decomposition_kwargs)

        if bias and linear.bias is not None:
            instance.bias.data = linear.bias.data

        return instance

    @classmethod
    def from_linear_list(cls, linear_list, in_tensorized_features, out_tensorized_features, rank, bias=True,
                         factorization='CP', decomposition_kwargs=dict(init='random')):
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
        if factorization == 'TTM' and len(linear_list) > 1:
            raise ValueError(f'TTM factorization only support single factorized layers but got {len(linear_list)} layers.')

        for linear in linear_list:
            out_features, in_features = linear.weight.shape
            assert(out_features == np.prod(out_tensorized_features))
            assert(in_features == np.prod(in_tensorized_features))

        instance = cls(in_tensorized_features, out_tensorized_features, bias=bias,
                       factorization=factorization, rank=rank, n_layers=len(linear_list),
                       device=linear.weight.device, dtype=linear.weight.dtype)
        weight_tensor = torch.stack([layer.weight.data for layer in linear_list])
        instance.weight.init_from_matrix(weight_tensor, **decomposition_kwargs)

        if bias:
            for i, layer in enumerate(linear_list):
                if layer.bias is not None:
                    instance.bias.data[i] = layer.bias.data
                    instance.has_bias[i] = 1

        return instance

    def __repr__(self):
        msg = (f'{self.__class__.__name__}(in_features={self.in_features}, out_features={self.out_features},'
               f' weight of size ({self.out_features}, {self.in_features}) tensorized to ({self.out_tensorized_features}, {self.in_tensorized_features}),'
               f'factorization={self.weight._name}, rank={self.rank}')
        if self.bias is None:
            msg += f', bias=False'

        if self.n_layers == 1:
            msg += ', with a single layer parametrized, '
            return msg

        msg += f' with {self.n_layers} layers jointly parametrized.'

        return msg


class SubFactorizedLinear(nn.Module):
    """Class representing one of the convolutions from the mother joint factorized convolution

    Parameters
    ----------

    Notes
    -----
    This relies on the fact that nn.Parameters are not duplicated:
    if the same nn.Parameter is assigned to multiple modules, they all point to the same data, 
    which is shared.
    """
    def __init__(self, main_linear, indices):
        super().__init__()
        self.main_linear = main_linear
        self.indices = indices

    def forward(self, x):
        return self.main_linear(x, self.indices)

    def extra_repr(self):
        msg = f'in_features={self.main_linear.in_features}, out_features={self.main_linear.out_features}'
        if self.main_linear.has_bias[self.indices]:
            msg += ', bias=True'
        return msg

    def __repr__(self):
        msg = f' {self.__class__.__name__} {self.indices} from main factorized layer.'
        msg += f'\n{self.__class__.__name__}('
        msg += self.extra_repr()
        msg += ')'
        return msg
