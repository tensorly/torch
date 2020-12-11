import pytest

from .._tensor_lasso import TuckerL1Regularizer, TTL1Regularizer, CPL1Regularizer
from .._trl import TuckerTRL, CPTRL, TensorTrainTRL

import tensorly as tl
tl.set_backend('pytorch')
from tensorly.testing import assert_
import numpy as np
import torch

@pytest.mark.parametrize('TRL, TensorL1Regularizer', 
                        [(TuckerTRL, TuckerL1Regularizer), 
                         (TensorTrainTRL, TTL1Regularizer),
                         (CPTRL, CPL1Regularizer)
                        ])
def test_tensor_lasso(TRL, TensorL1Regularizer):
    in_shape = (5, 5)
    out_shape = (6, )
    batch_size = 2
    device = 'cpu'
    penalty = 0.15
    input_size = (batch_size,) + in_shape
    data = torch.tensor(np.random.random(input_size), dtype=torch.float32).to(device)

    Regularizer = TensorL1Regularizer(penalty=penalty, normalize_loss=False)
    trl = TRL(in_shape, in_shape, rank='same')
    trl2 = TRL(in_shape, out_shape, rank='same')

    if isinstance(trl, CPTRL):
        # In case the CP tensor is normalized
        trl.weights.data.fill_(1)
        trl2.weights.data.fill_(1)

    Regularizer.apply(trl)
    Regularizer.apply(trl2)

    data = trl(data)
    l1 = Regularizer.loss
    data = trl2(data)
    l2 = Regularizer.loss

    tol = 1e-5
    sum_rank = trl.rank if isinstance(trl.rank, int) else sum(trl.rank)
    assert_(l1 - sum_rank*penalty < tol)
    sum_rank = trl2.rank if isinstance(trl2.rank, int) else sum(trl2.rank)
    assert_(l2 - (l1 + sum_rank*penalty) < tol)

    Regularizer.remove(trl2)
    assert_(not trl2._decomposition_forward_pre_hooks)

    # Set the regularization weights to 0
    if isinstance(trl, CPTRL):
        trl.weights.data.zero_()
    else:
        for i in range(len(trl.lasso_weights)):
            trl.lasso_weights[i].data.zero_()
    Regularizer.reset()

    data = torch.tensor(np.random.random(input_size), dtype=torch.float32).to(device)
    # Regularization weights are all zero so output should be zero too
    data = trl(data)
    assert_(Regularizer.loss == 0)
    assert_(not data.to(bool).any())

    # Create a TRL with weights of 1 for the CP case 
    trl = TRL(in_shape, in_shape, rank='same')
    if isinstance(trl, CPTRL):
        # In case the CP tensor is normalized
        trl.weights.data.fill_(1)
    
    # Check that the loss is properly normalized between 0 and 1
    Regularizer = TensorL1Regularizer(penalty=1, normalize_loss=True)
    Regularizer.apply(trl)
    data = trl(data)
    l1 = Regularizer.loss
    assert(abs(l1 - 1) < 1e-5)
    Regularizer.remove(trl)

    # Check that the weights are correctly clamped
    Regularizer = TensorL1Regularizer(penalty=1, clamp_weights=1, normalize_loss=True)
    Regularizer.apply(trl)
    # Set the lasso weight to a value > 1
    if isinstance(trl, CPTRL):
        trl.weights.data.fill_(2)
    else:
        for i in range(len(trl.lasso_weights)):
            trl.lasso_weights[i].data.fill_(2)
    # Apply lasso in forward
    data = trl(data)
    # Check that the lasso weights were correctly clamped
    if isinstance(trl, CPTRL):
        assert_(int(torch.sum(trl.weights > 1)) == 0)
    else:
        for i in range(len(trl.lasso_weights)):
            assert_(int(torch.sum(trl.lasso_weights[i] > 1)) == 0)


    # Finally, check that the loss is properly thresholded
    Regularizer = TensorL1Regularizer(penalty=1, clamp_weights=1, threshold=0.5, normalize_loss=True)
    Regularizer.apply(trl)
    # Set the lasso weight to a value > threshold
    if isinstance(trl, CPTRL):
        trl.weights.data.fill_(0.45)
    else:
        for i in range(len(trl.lasso_weights)):
            trl.lasso_weights[i].data.fill_(0.2)
    # Apply lasso in forward
    data = trl(data)
    # Check that the lasso weights were correctly thresholded
    if isinstance(trl, CPTRL):
        assert_(torch.sum(torch.nonzero(trl.weights).to(bool)) == 0)
    else:
        for i in range(len(trl.lasso_weights)):
            assert_(torch.sum(torch.nonzero(trl.lasso_weights[i]).to(bool)) == 0)
