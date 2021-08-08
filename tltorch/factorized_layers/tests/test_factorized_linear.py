import pytest 
from torch import nn
from ..factorized_linear import FactorizedLinear
from ... import TensorizedTensor

import tensorly as tl
tl.set_backend('pytorch')
from tensorly import testing

@pytest.mark.parametrize('factorization', ['CP', 'Tucker', 'BlockTT'])
def test_FactorizedLinear(factorization):
    random_state = 12345
    rng = tl.check_random_state(random_state)
    batch_size = 2
    in_features = 9
    in_shape = (3, 3)
    out_features = 16
    out_shape = (4, 4)
    data = tl.tensor(rng.random_sample((batch_size, in_features)))

    # Creat from a tensor factorization
    tensor = TensorizedTensor.new((out_shape, in_shape), rank='same', factorization=factorization)
    tensor.normal_()
    fc = nn.Linear(in_features, out_features, bias=True)
    fc.weight.data = tensor.to_matrix()
    tfc = FactorizedLinear(in_shape, out_shape, rank='same', factorization=tensor, bias=True)
    tfc.bias.data = fc.bias
    res_fc = fc(data)
    res_tfc = tfc(data)
    testing.assert_array_almost_equal(res_fc, res_tfc, decimal=2)

    # Decompose an existing layer
    fc = nn.Linear(in_features, out_features, bias=True)
    tfc = FactorizedLinear.from_linear(fc, (3, 3), (4, 4), rank=34, bias=True)
    res_fc = fc(data)
    res_tfc = tfc(data)
    testing.assert_array_almost_equal(res_fc, res_tfc, decimal=2)

    # Multi-layer factorization
    fc1 = nn.Linear(in_features, out_features, bias=True)
    fc2 = nn.Linear(in_features, out_features, bias=True)
    tfc = FactorizedLinear.from_linear_list([fc1, fc2], in_shape, out_shape, rank=38, bias=True)
    ## Test first parametrized conv
    res_fc = fc1(data)
    res_tfc = tfc[0](data)
    testing.assert_array_almost_equal(res_fc, res_tfc, decimal=2)
    ## Test second parametrized conv
    res_fc = fc2(data)
    res_tfc = tfc[1](data)
    testing.assert_array_almost_equal(res_fc, res_tfc, decimal=2)