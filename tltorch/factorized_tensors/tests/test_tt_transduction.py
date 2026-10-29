import pytest
import torch

from tltorch import FactorizedTensor


@pytest.mark.parametrize('mode', [0, 1, 3])
@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
@pytest.mark.parametrize('custom_factor', [False, True])
def test_tt_transduction_preserves_dtype_and_values(mode, dtype, custom_factor):
    tensor = FactorizedTensor.new((3, 4, 5), rank=2, factorization='TT')
    tensor.normal_().to(dtype=dtype)
    # Reconstruct independently of TensorLy's TT contraction.
    first, middle, last = tensor.factors
    dense = torch.einsum('ia,ajb,bk->ijk', first[0], middle, last[:, :, 0])
    scales = torch.tensor([0.5, 1.5], dtype=dtype) if custom_factor else torch.ones(2, dtype=dtype)
    new_factor = None
    if custom_factor:
        rank = tensor.rank[mode]
        new_factor = torch.eye(rank, dtype=dtype)[:, None, :] * scales[None, :, None]

    tensor.transduct(2, mode=mode, new_factor=new_factor)
    shape = list(dense.shape)
    shape.insert(mode, 2)
    scale_shape = [1] * len(shape)
    scale_shape[mode] = 2
    expected = dense.unsqueeze(mode).expand(shape) * scales.reshape(scale_shape)
    torch.testing.assert_close(tensor.to_tensor(), expected)
    assert all(factor.dtype == dtype for factor in tensor.factors)
    assert all(factor.device.type == 'cpu' for factor in tensor.factors)
    assert tensor.shape == tuple(shape)

    restored = FactorizedTensor.new(tensor.shape, rank=tensor.rank, factorization='TT', dtype=dtype)
    restored.load_state_dict(tensor.state_dict())
    torch.testing.assert_close(restored.to_tensor(), expected)
