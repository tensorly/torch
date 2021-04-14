import pytest

from .._tensor_lasso import TuckerLasso, TTLasso, CPLasso, remove_tensor_lasso, tensor_lasso
from ... import FactorizedTensor

import tensorly as tl
tl.set_backend('pytorch')
from tensorly import testing
import numpy as np
import torch

@pytest.mark.parametrize('factorization', ['cp', 'tucker', 'tt'])
def test_tensor_lasso(factorization):
    shape = (5, 5, 6)
    rank = 3
    tensor1 = FactorizedTensor.new(shape, rank, factorization=factorization).normal_()
    tensor2 = FactorizedTensor.new(shape, rank, factorization=factorization).normal_()

    lasso = tensor_lasso(factorization, penalty=1, clamp_weights=False, normalize_loss=False)

    lasso.apply(tensor1)
    lasso.apply(tensor2)

    # Sum of weights all equal to a given value
    value = 1.5
    lasso.set_weights(tensor1, value)
    lasso.set_weights(tensor2, value)

    data = torch.sum(tensor1())
    l1 = lasso.loss
    data = torch.sum(tensor2())
    l2 = lasso.loss

    # The result should be n-param * value
    # First tensor
    if factorization == 'tt':
        sum_rank = sum(tensor1.rank[1:-1])
    elif factorization == 'tucker':
        sum_rank = sum(tensor1.rank)
    elif factorization == 'cp':
        sum_rank = tensor1.rank
    testing.assert_(l1 == sum_rank*value)
    # Second tensor lasso
    if factorization == 'tt':
        sum_rank += sum(tensor2.rank[1:-1])
    elif factorization == 'tucker':
        sum_rank += sum(tensor2.rank)
    elif factorization == 'cp':
        sum_rank += tensor2.rank
    testing.assert_(l2 == sum_rank*value)

    testing.assert_(tensor1._forward_hooks)
    testing.assert_(tensor2._forward_hooks)

    ### Test when all weights are 0
    lasso.reset()
    lasso.set_weights(tensor1, 0)
    lasso.set_weights(tensor2, 0)
    torch.sum(tensor1()) + torch.sum(tensor2())
    testing.assert_(lasso.loss == 0)
    
    # Check the Lasso correctly removed
    remove_tensor_lasso(tensor1)
    testing.assert_(not tensor1._forward_hooks)
    testing.assert_(tensor2._forward_hooks)
    remove_tensor_lasso(tensor2)
    testing.assert_(not tensor1._forward_hooks)

    # Check normalization between 0 and 1
    lasso = tensor_lasso(factorization, penalty=1, normalize_loss=True, clamp_weights=False)
    lasso.apply(tensor1)
    tensor1()
    l1 = lasso.loss
    assert(abs(l1 - 1) < 1e-5)
    remove_tensor_lasso(tensor1)