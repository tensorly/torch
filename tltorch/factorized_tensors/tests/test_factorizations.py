import numpy as np
import pytest
from tensorly.testing import assert_array_almost_equal
import torch
from torch import testing

from ..factorized_tensor import FactorizedTensor, CPTensor, TuckerTensor, TTTensor

@pytest.mark.parametrize('factorization', ['CP', 'Tucker', 'TT'])
def test_FactorizedTensor(factorization):
    """Test for FactorizedTensor"""
    shape = (4, 3, 2, 5)
    fact_tensor = FactorizedTensor.new(shape=shape, rank='same', factorization=factorization)
    fact_tensor.normal_()

    # Check that the correct type of factorized tensor is created
    assert fact_tensor._name.lower() == factorization.lower()
    mapping = dict(CP=CPTensor, Tucker=TuckerTensor, TT=TTTensor)
    assert isinstance(fact_tensor, mapping[factorization])

    # Check the shape of the factorized tensor and reconstruction
    reconstruction = fact_tensor.to_tensor()
    assert fact_tensor.shape == reconstruction.shape == shape

    # Check that indexing the factorized tensor returns the correct result
    # np_s converts intuitive array indexing to proper tuples
    indices = [
        np.s_[:, 2, :], # = (slice(None), 2, slice(None))
        np.s_[2:, :2, :, 0],
        np.s_[0, 2, 1, 3],
        np.s_[-1, :, :, ::2]
    ]
    for idx in indices:
        assert reconstruction[idx].shape == fact_tensor[idx].shape
        res = fact_tensor[idx]
        if not torch.is_tensor(res):
            res = res.to_tensor()
        testing.assert_allclose(reconstruction[idx], res)


@pytest.mark.parametrize('factorization', ['CP', 'TT'])
def test_transduction(factorization):
    """Test for transduction"""
    shape = (3, 4, 5)
    new_dim = 2
    for mode in range(3):
        fact_tensor = FactorizedTensor.new(shape=shape, rank=6, factorization=factorization)
        fact_tensor.normal_()
        original_rec = fact_tensor.to_tensor()
        fact_tensor = fact_tensor.transduct(new_dim, mode=mode)
        rec = fact_tensor.to_tensor()
        true_shape = list(shape); true_shape.insert(mode, new_dim)
        assert tuple(fact_tensor.shape) == tuple(rec.shape) == tuple(true_shape)

        indices = [slice(None)]*mode
        for i in range(new_dim):
            testing.assert_allclose(original_rec, rec[tuple(indices + [i])])
