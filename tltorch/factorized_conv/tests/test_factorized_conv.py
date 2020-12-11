from tltorch.factorized_conv import Conv1D, CPConv
from tltorch.factorized_conv import TuckerConv, TTConv
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
import numpy as np
import math

import tensorly as tl
from tensorly.decomposition import parafac
from tensorly import random
from tensorly import tenalg
tl.set_backend('pytorch')

from tensorly.testing import assert_array_almost_equal
import pytest


def single_conv_test(FactorizedConv, implementation, random_tensor_generator, reconstruction_fun,
              order=2, rank=None, rng=None, input_channels=2, output_channels=4,
              kernel_size=3, batch_size=1, activation_size=8, device='cpu'):
    rng = random.check_random_state(rng)
    input_shape = (batch_size, input_channels) + (activation_size, )*order
    kernel_shape = (output_channels, input_channels) + (kernel_size, )*order

    if rank is None:
        rank = max(kernel_shape)
    
    if order == 1:
        FullConv = nn.Conv1d
    elif order == 2:
        FullConv = nn.Conv2d
    elif order == 3:
        FullConv = nn.Conv3d

    # random input tensor
    decomposed_weights = random_tensor_generator(kernel_shape, rank=rank, full=False)
    full_weights = reconstruction_fun(decomposed_weights).to(device)
    data = torch.tensor(rng.random_sample(input_shape), dtype=torch.float32).to(device)

    conv = FullConv(input_channels, output_channels, kernel_size, bias=True, padding=1)
    conv.weight.data = full_weights
    true_bias = conv.bias.data
    true_res = conv(data)

    fact_conv = FactorizedConv(input_channels, output_channels, kernel_size, order=order, 
                               implementation=implementation, rank=rank, bias=True, padding=1)
    # First check it has the current type
    msg = f'Created implementation={implementation} but {fact_conv.implementation} was created.'
    assert fact_conv.implementation == implementation, msg

    # Check that it gives the same result as the full conv
    print('original', conv.bias.shape)
    fact_conv.init_from_decomposition(decomposed_weights, bias=conv.bias)
    print('original', fact_conv.bias.shape)
    res = fact_conv(data)    
    msg = f'{fact_conv.__class__.__name__} does not give same result as {FullConv.__class__.__name__}.'
    assert_array_almost_equal(true_res, res, decimal=4, err_msg=msg)
    
    # Check that the parameters of the decomposition are transposed back correctly
    decomposed_weights, bias = fact_conv.get_decomposition(return_bias=True)
    rec = reconstruction_fun(decomposed_weights)
    msg = msg = f'{fact_conv.__class__.__name__} does not return the decomposition it was constructed with.'
    assert_array_almost_equal(rec, full_weights, err_msg=msg)
    msg = msg = f'{fact_conv.__class__.__name__} does not return the bias it was constructed with.'
    assert_array_almost_equal(bias, true_bias, err_msg=msg)


@pytest.mark.parametrize('order', [1, 2, 3])
@pytest.mark.parametrize('implementation', ['reconstructed', 'factorized', 'mobilenet'])
def test_CPConv(implementation, order):
    """Test for Factorized CPConv"""
    single_conv_test(CPConv, implementation, random.random_cp, tl.cp_to_tensor, order=order, rank=5)
    
@pytest.mark.parametrize('order', [1, 2, 3])
@pytest.mark.parametrize('implementation', ['reconstructed', 'factorized'])
def test_TuckerConv(implementation, order):
    """Test for Factorized TuckerConv"""
    rank = (2, 4) + (3, )*order
    single_conv_test(TuckerConv, implementation, random.random_tucker, tl.tucker_to_tensor, order=order, rank=rank)

@pytest.mark.parametrize('order', [1, 2, 3])
@pytest.mark.parametrize('implementation', ['reconstructed', 'factorized'])
def test_TTConv(implementation, order):
    """Test for Factorized TuckerConv"""
    rank = (1, 4, 4) + (3, )*(order-1) + (1, )
    
    def random_tensor_generator(shape, rank, full=False):
        shape = list(shape)
        out_channel = shape.pop(0)
        shape += [out_channel]
        shape = tuple(shape)
        return random.random_tt(shape, rank, full=full)

    def reconstruction_fun(tt_tensor):
        device = tt_tensor[0].device
        return tl.moveaxis(tl.tt_to_tensor(tt_tensor).to(device), -1, 0)

    single_conv_test(TTConv, implementation, random_tensor_generator, reconstruction_fun, order=order, rank=rank)
