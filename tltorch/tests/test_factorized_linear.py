import pytest 
from torch import nn
from .._factorized_linear import TTLinear, CPLinear, TuckerLinear, TTMLinear

import tensorly as tl
tl.set_backend('pytorch')
from tensorly import random
from tensorly import testing

@pytest.mark.parametrize('FactorizedLinear', [TuckerLinear, CPLinear, TTLinear, TTMLinear])
def test_FactorizedLinear(FactorizedLinear):
    random_state = 12345
    rng = random.check_random_state(random_state)
    batch_size = 2
    in_features = 9
    out_features = 16

    data = tl.tensor(rng.random_sample((batch_size, in_features)))
    fc = nn.Linear(in_features, out_features, bias=True)
    tfc = FactorizedLinear.from_linear(fc, (3, 3, 4, 4), rank=34, bias=True)
    res_fc = fc(data)
    res_tfc = tfc(data)

    testing.assert_array_almost_equal(res_fc, res_tfc, decimal=2)