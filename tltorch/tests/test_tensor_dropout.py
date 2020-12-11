import pytest

from .._tensor_dropout import cp_dropout, remove_cp_dropout
from .._tensor_dropout import tt_dropout, remove_tt_dropout
from .._tensor_dropout import tucker_dropout, remove_tucker_dropout

from .._trl import TuckerTRL, CPTRL, TensorTrainTRL

import tensorly as tl
tl.set_backend('pytorch')

def test_tucker_dropout():
    """Test for Tucker Dropout"""
    in_shape = (10, 10)
    out_shape = (10, )
    rank = (7, 8, 9)
    trl = TuckerTRL(in_shape, out_shape, rank=rank)
    trl = tucker_dropout(trl, 1)
    core, _ = trl._process_decomposition()
    assert (tl.shape(core) == (1, 1, 1))

    remove_tucker_dropout(trl)
    assert (not trl._decomposition_forward_pre_hooks)

    trl = tucker_dropout(trl, 0)
    core, _ = trl._process_decomposition()
    assert (tl.shape(core) == rank)

def test_cp_dropout():
    """Test for CP Dropout"""
    in_shape = (10, 10)
    out_shape = (10, )
    rank = 8
    trl = CPTRL(in_shape, out_shape, rank=rank)
    trl = cp_dropout(trl, 1)
    weights, _ = trl._process_decomposition()
    assert (len(weights) == (1))

    remove_cp_dropout(trl)
    assert (not trl._decomposition_forward_pre_hooks)

    trl = cp_dropout(trl, 0)
    weights, _ = trl._process_decomposition()
    assert (len(weights) == rank)


def test_tt_dropout():
    """Test for TT Dropout"""
    in_shape = (10, 10)
    out_shape = (10, )
    # Use the same rank for all factors
    rank = 4
    trl = TensorTrainTRL(in_shape, out_shape, rank=rank)
    trl = tt_dropout(trl, 1)
    factors = trl._process_decomposition()
    for f in factors:
        assert (f.shape[0] == f.shape[-1] == 1)

    remove_tt_dropout(trl)
    assert (not trl._decomposition_forward_pre_hooks)

    trl = tt_dropout(trl, 0)
    factors = trl._process_decomposition()
    for i, f in enumerate(factors):
        if i:
            assert (f.shape[0] == rank)
        else: # boundary conditions: first and last rank are equal to 1
            assert (f.shape[-1] == rank)