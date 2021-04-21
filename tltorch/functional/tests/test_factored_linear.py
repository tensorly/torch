import torch
import tensorly as tl
import tensorly.random as tlr
from ..factorized_linear import tt_factorized_linear
from tensorly.testing import assert_array_almost_equal

tl.set_backend('pytorch')

def test_tt_factorized_linear():
    x = tlr.random_tt((3,5,4), rank=[1,2,4,1])
    weights = tlr.random_tt_matrix((3,5,4,3,5,4), rank=[1,2,4,1])
    out = tt_factorized_linear(x, weights)
    out = tl.reshape(out, (-1, 1))
    manual_out = tl.dot(weights.to_matrix(), tl.reshape(x.to_tensor(), (-1, 1)))
    assert_array_almost_equal(out, manual_out, decimal=4)
