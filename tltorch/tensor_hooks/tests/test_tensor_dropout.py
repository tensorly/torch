from ... import FactorizedTensor
from ...factorized_layers import TRL
from .._tensor_dropout import tensor_dropout, remove_tensor_dropout

import tensorly as tl
tl.set_backend('pytorch')
    
def test_tucker_dropout():
    """Test for Tucker Dropout"""
    shape = (10, 11, 12)
    rank = (7, 8, 9)
    tensor = FactorizedTensor.new(shape, rank=rank, factorization='Tucker')
    tensor = tensor_dropout(tensor, 0.999)
    core = tensor().core
    assert (tl.shape(core) == (1, 1, 1))

    remove_tensor_dropout(tensor)
    assert (not tensor._forward_hooks)

    tensor = tensor_dropout(tensor, 0)
    core = tensor().core
    assert (tl.shape(core) == rank)

def test_cp_dropout():
    """Test for CP Dropout"""
    shape = (10, 11, 12)
    rank = 8
    tensor = FactorizedTensor.new(shape, rank=rank, factorization='CP')
    tensor = tensor_dropout(tensor, 0.999)
    weights = tensor().weights
    assert (len(weights) == (1))

    remove_tensor_dropout(tensor)
    assert (not tensor._forward_hooks)

    tensor = tensor_dropout(tensor, 0)
    weights = tensor().weights
    assert (len(weights) == rank)


def test_tt_dropout():
    """Test for TT Dropout"""
    shape = (10, 11, 12)
    # Use the same rank for all factors
    rank = 4
    tensor = FactorizedTensor.new(shape, rank=rank, factorization='TT')
    tensor = tensor_dropout(tensor, 0.999)
    factors = tensor().factors
    for f in factors:
        assert (f.shape[0] == f.shape[-1] == 1)

    remove_tensor_dropout(tensor)
    assert (not tensor._forward_hooks)

    tensor = tensor_dropout(tensor, 0)
    factors = tensor().factors
    for i, f in enumerate(factors):
        if i:
            assert (f.shape[0] == rank)
        else: # boundary conditions: first and last rank are equal to 1
            assert (f.shape[-1] == rank)
