import pytest
import torch
from torch import nn

import tensorly as tl
tl.set_backend('pytorch')
from ... import FactorizedTensor
from tltorch.factorized_layers import FactorizedConv
from tensorly.testing import assert_array_almost_equal

from ..factorized_convolution import (FactorizedConv, kernel_shape_to_factorization_shape, tensor_to_kernel)

@pytest.mark.parametrize('factorization, implementation', 
        [('CP', 'factorized'), ('CP', 'reconstructed'), ('CP', 'mobilenet'),
         ('Tucker', 'factorized'), ('Tucker', 'reconstructed'),
         ('TT', 'factorized'), ('TT', 'reconstructed')])
def test_single_conv(factorization, implementation,
                     order=2, rank=0.5, rng=None, input_channels=4, output_channels=5,
                     kernel_size=3, batch_size=1, activation_size=(8, 7), device='cpu'):
    rng = tl.check_random_state(rng)
    input_shape = (batch_size, input_channels) + activation_size
    kernel_shape = (output_channels, input_channels) + (kernel_size, )*order

    if rank is None:
        rank = max(kernel_shape)

    if order == 1:
        FullConv = nn.Conv1d
    elif order == 2:
        FullConv = nn.Conv2d
    elif order == 3:
        FullConv = nn.Conv3d

    # Factorized input tensor
    factorization_shape = kernel_shape_to_factorization_shape(factorization, kernel_shape)
    decomposed_weights = FactorizedTensor.new(shape=factorization_shape, rank=rank, factorization=factorization).normal_(0, 1)
    full_weights = tensor_to_kernel(factorization, decomposed_weights.to_tensor().to(device))
    data = torch.tensor(rng.random_sample(input_shape), dtype=torch.float32).to(device)

    # PyTorch regular Conv
    conv = FullConv(input_channels, output_channels, kernel_size, bias=True, padding=1)
    true_bias = conv.bias.data
    conv.weight.data = full_weights

    # Factorized conv
    fact_conv = FactorizedConv.from_factorization(decomposed_weights, implementation=implementation,
                                                  bias=true_bias, padding=1)

    # First check it has the correct implementation
    msg = f'Created implementation={implementation} but {fact_conv.implementation} was created.'
    assert fact_conv.implementation == implementation, msg

    # Check that it gives the same result as the full conv
    true_res = conv(data)
    res = fact_conv(data) 
    msg = f'{fact_conv.__class__.__name__} does not give same result as {FullConv.__class__.__name__}.'
    assert_array_almost_equal(true_res, res, decimal=4, err_msg=msg)

    # Check that the parameters of the decomposition are transposed back correctly
    decomposed_weights, bias = fact_conv.weight, fact_conv.bias
    rec = tensor_to_kernel(factorization, decomposed_weights.to_tensor())
    msg = msg = f'{fact_conv.__class__.__name__} does not return the decomposition it was constructed with.'
    assert_array_almost_equal(rec, full_weights, err_msg=msg)
    msg = msg = f'{fact_conv.__class__.__name__} does not return the bias it was constructed with.'
    assert_array_almost_equal(bias, true_bias, err_msg=msg)

    conv = FullConv(input_channels, output_channels, kernel_size, bias=True, padding=1)
    conv.weight.data.uniform_(-1, 1)
    fact_conv = FactorizedConv.from_conv(conv, rank=30, factorization=factorization)#, decomposition_kwargs=dict(init='svd', l2_reg=1e-5))
    true_res = conv(data)
    res = fact_conv(data) 
    msg = f'{fact_conv.__class__.__name__} does not give same result as {FullConv.__class__.__name__}.'
    assert_array_almost_equal(true_res, res, decimal=2, err_msg=msg)
