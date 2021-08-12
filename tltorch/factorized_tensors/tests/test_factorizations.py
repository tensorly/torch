import numpy as np
import pytest
import math
import torch
from torch import testing

from tltorch.factorized_tensors.tensorized_matrices import CPTensorized, TuckerTensorized, BlockTT
from tltorch.factorized_tensors.core import TensorizedTensor

from ..factorized_tensors import FactorizedTensor, CPTensor, TuckerTensor, TTTensor

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


@pytest.mark.parametrize('factorization', ['BlockTT', 'CP']) #['CP', 'Tucker', 'BlockTT'])
@pytest.mark.parametrize('batch_size', [(), (4,)])
def test_TensorizedMatrix(factorization, batch_size):
    """Test for TensorizedMatrix"""
    row_tensor_shape = (4, 3, 2)
    column_tensor_shape = (5, 3, 2)
    row_shape = math.prod(row_tensor_shape)
    column_shape = math.prod(column_tensor_shape)
    tensor_shape = batch_size + (row_tensor_shape, column_tensor_shape)

    fact_tensor = TensorizedTensor.new(tensor_shape,
                                       rank=0.5, factorization=factorization)
    fact_tensor.normal_()

    # Check that the correct type of factorized tensor is created
    assert fact_tensor._name.lower() == factorization.lower()
    mapping = dict(CP=CPTensorized, Tucker=TuckerTensorized, BlockTT=BlockTT)
    assert isinstance(fact_tensor, mapping[factorization])

    # Check that the matrix has the right shape
    reconstruction = fact_tensor.to_matrix()
    if batch_size:
        assert fact_tensor.shape[1] == row_shape == reconstruction.shape[1]
        assert fact_tensor.shape[2] == column_shape == reconstruction.shape[2]
        assert fact_tensor.ndim == 3
    else:
        assert fact_tensor.shape[0] == row_shape == reconstruction.shape[0]
        assert fact_tensor.shape[1] == column_shape == reconstruction.shape[1]
        assert fact_tensor.ndim == 2

    # Check that indexing the factorized tensor returns the correct result
    # np_s converts intuitive array indexing to proper tuples
    indices = [
        np.s_[:, :], # = (slice(None), slice(None))
        np.s_[:, 2],
        np.s_[2, 3],
        np.s_[1, :]
    ]
    for idx in indices:
        assert tuple(reconstruction[idx].shape) == tuple(fact_tensor[idx].shape)
        res = fact_tensor[idx]
        if not torch.is_tensor(res):
            res = res.to_matrix()
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

@pytest.mark.parametrize('unsqueezed_init', ['average', 1.2])
def test_tucker_init_unsqueezed_modes(unsqueezed_init):
    """Test for Tucker Factorization init from tensor with unsqueezed_modes
    """
    tensor = FactorizedTensor.new((4, 4, 4), rank=(4, 1, 4), factorization='tucker')
    mat = torch.randn((4, 4))
    
    tensor.init_from_tensor(mat, unsqueezed_modes=[1], unsqueezed_init=unsqueezed_init)
    rec = tensor.to_tensor()

    if unsqueezed_init == 'average':
        coef = 1/4
    else:
        coef = unsqueezed_init

    for i in range(4):
        testing.assert_allclose(rec[:, i], mat*coef)
