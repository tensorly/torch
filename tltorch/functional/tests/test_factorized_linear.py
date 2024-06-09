from ...factorized_tensors import TensorizedTensor
from ..factorized_linear import linear_tucker, linear_blocktt, linear_cp
import torch

import tensorly as tl
tl.set_backend('pytorch')
from tensorly import testing
from math import prod

import pytest

# Author: Jean Kossaifi


@pytest.mark.parametrize('factorization, factorized_linear', 
    [('tucker', linear_tucker), ('blocktt', linear_blocktt), ('cp', linear_cp)])
def test_linear_tensor_dot_tucker(factorization, factorized_linear):
    in_shape = (4, 5)
    in_dim = prod(in_shape)
    out_shape = (6, 2)
    rank = 3
    batch_size = 2

    tensor = tl.randn((batch_size, in_dim), dtype=tl.float32)
    fact_weight = TensorizedTensor.new((out_shape, in_shape), rank=rank,
                                       factorization=factorization)
    fact_weight.normal_()
    full_weight = fact_weight.to_matrix()
    true_res = torch.matmul(tensor, full_weight.T)
    res = factorized_linear(tensor, fact_weight, transpose=True)
    res = res.reshape(batch_size, -1)
    testing.assert_array_almost_equal(true_res, res, decimal=5)
    
