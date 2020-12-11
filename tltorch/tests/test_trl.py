import torch
from torch.utils import data
from torch import nn

from tltorch import TuckerTRL, CPTRL, TensorTrainTRL

import tensorly as tl
tl.set_backend('pytorch')
from tensorly import tenalg
from tensorly import random 
from tensorly import testing

import pytest


def optimize_trl(trl, loader, lr=0.05, n_epoch=100, verbose=False):
    """Function that takes as input a TRL, dataset and optimizes the TRL

    Parameters
    ----------
    trl : tltorch.TRL
    loader : Pytorch dataset, returning batches (batch, labels)
    lr : float, default is 0.1
        learning rate
    n_epoch : int, default is 100
    verbose : bool, default is False
        level of verbosity

    Returns
    -------
    (trl, objective function loss)
    """
    optimizer = torch.optim.Adam(trl.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=20, cooldown=10, verbose=verbose)

    for epoch in range(n_epoch):
        for (sample_batch, label_batch) in loader:

            # Important: do not forget to reset the gradients
            optimizer.zero_grad()

            # Reconstruct the tensor from the decomposed form
            pred = trl(sample_batch)

            # squared l2 loss
            loss = tl.norm(pred - label_batch, 2)

            loss.backward()
            optimizer.step()
            scheduler.step(loss)

        if epoch % 10 == 0:
            if verbose:
                print(f"Epoch {epoch},. loss: {loss.item()}")
    
    return trl, loss.item()


@pytest.mark.parametrize('TRL, random_tensor, true_rank, rank',
                         [(TuckerTRL, random.random_tucker, (3, 2, 3, 2, 1, 2), (4, 3, 3, 2, 1, 2)), #(5, 5, 5, 4, 1, 3)),
                          (CPTRL, random.random_cp, 3, 6),
                          (TensorTrainTRL, random.random_tt, (1, 3, 2, 2, 2, 2, 1), (1, 5, 4, 4, 4, 4, 1))])
def test_trl(TRL, random_tensor, true_rank, rank):
    """Test for the TRL
    """
    # Parameter of the experiment
    input_shape = (5, 4, 2)
    output_shape = (2, 2, 2)
    batch_size = 500

    # fix the random seed for reproducibility
    random_state = 12345

    rng = random.check_random_state(random_state)
    tol = 0.08

    # Generate a random tensor
    samples = tl.tensor(rng.normal(size=(batch_size, *input_shape), loc=0, scale=1))
    true_bias = tl.tensor(rng.uniform(size=output_shape))

    true_weight = random_tensor(shape=input_shape+output_shape, rank=true_rank, full=True, random_state=rng)
    labels = tenalg.inner(samples, true_weight, n_modes=len(input_shape)) + true_bias

    dataset = data.TensorDataset(samples, labels)
    loader = data.DataLoader(dataset, batch_size=32)
    
    trl = TRL(input_shape=input_shape, output_shape=output_shape, rank=rank, bias=True)
    print(f'Testing {trl.__class__.__name__}.')
    #trl.init_from_random(decompose_full_weight=True)
    trl, _ = optimize_trl(trl, loader, verbose=False)

    with torch.no_grad():
        rec_weights = trl.full_weight()
        rec_loss = tl.norm(rec_weights - true_weight)/tl.norm(true_weight)

    with torch.no_grad():
        bias_rec_loss = tl.norm(trl.bias - true_bias)/tl.norm(true_bias)

    testing.assert_(rec_loss <= tol, msg=f'Rec_loss of the weights={rec_loss} higher than tolerance={tol}')
    testing.assert_(bias_rec_loss <= tol, msg=f'Rec_loss of the bias={bias_rec_loss} higher than tolerance={tol}')

@pytest.mark.parametrize('order', [2, 3])
@pytest.mark.parametrize('project_input', [False, True])
@pytest.mark.parametrize('learn_pool', [True, False])
def test_TuckerTRL(order, project_input, learn_pool):
    """Test for Tucker TRL

        Here, we test specifically for init from fully-connected layer
            (both when learning the pooling and not).
        
        We also test that projecting the input or not doesn't change the results
    """
    in_features = 10
    out_features = 12
    batch_size = 2
    spatial_size = 4
    in_rank = 10
    out_rank = 12

    # fix the random seed for reproducibility and create random input
    random_state = 12345
    rng = random.check_random_state(random_state)
    data = tl.tensor(rng.random_sample((batch_size, in_features) + (spatial_size, )*order))

    # Build a simple net with avg-pool, flatten + fully-connected
    if order == 2:
        pool = nn.AdaptiveAvgPool2d((1, 1))
    else:
        pool = nn.AdaptiveAvgPool3d((1, 1, 1))
    fc = nn.Linear(in_features, out_features, bias=False)

    def net(data):
        x = pool(data)
        x = x.squeeze()
        x = fc(x)
        return x

    res_fc = net(tl.copy(data))

    # A replacement TRL
    out_shape = (out_features, )
    if learn_pool:
        # Learn the average pool as part of the TRL
        in_shape = (in_features, ) + (spatial_size, )*order
        rank = (in_rank, ) + (1, )*order + (out_rank, )
        pooling_modes = list(range(1, order+1))
    else:
        in_shape = (in_features, )
        rank = (in_rank, out_rank)
        pooling_modes=None
        data = pool(data).squeeze()
    trl = TuckerTRL(in_shape, out_shape, rank, project_input=project_input)
    trl.init_from_linear(fc.weight, None, pooling_modes=pooling_modes)
    res_trl = trl(data)

    testing.assert_array_almost_equal(res_fc, res_trl)
    
@pytest.mark.parametrize('TRL', [TuckerTRL, CPTRL, TensorTrainTRL])
@pytest.mark.parametrize('bias', [True, False])
def test_TRL_from_linear(TRL, bias):
    """Test for CP and TT TRL

        Here, we test specifically for init from fully-connected layer
    """
    in_features = 10
    out_features = 12
    batch_size = 2

    # fix the random seed for reproducibility and create random input
    random_state = 12345
    rng = random.check_random_state(random_state)
    data = tl.tensor(rng.random_sample((batch_size, in_features)))
    fc = nn.Linear(in_features, out_features, bias=bias)
    res_fc = fc(tl.copy(data))
    trl = TRL((in_features, ), (out_features, ), rank=10, bias=bias)
    trl.init_from_linear(fc.weight, fc.bias)
    res_trl = trl(data)

    testing.assert_array_almost_equal(res_fc, res_trl, decimal=2)
    
