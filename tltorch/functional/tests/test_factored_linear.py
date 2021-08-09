import torch
import tensorly as tl
import tensorly.random as tlr
from ..factorized_linear import tt_factorized_linear
from tensorly.testing import assert_array_almost_equal

tl.set_backend('pytorch')

def test_tt_factorized_linear():
    x = tlr.random_tt((2,7), rank=[1,7,1])
    weights = tlr.random_tt_matrix((2,7,2,7), rank=[1,10,1])
    out = tt_factorized_linear(x, weights).to_tensor()
    out = tl.reshape(out, (-1, 1))
    manual_out = tl.dot(weights.to_matrix(), tl.reshape(x.to_tensor(), (-1, 1)))
    assert_array_almost_equal(out, manual_out, decimal=4)
