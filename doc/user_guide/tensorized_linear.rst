Tensorized Linear Layers
========================

Linear layers are parametrized by matrices. However, it is possible to
*tensorize* them, i.e. reshape them into higher-order tensors in order
to compress them.

You can do this easily in TensorLy-Torch:

.. code-block:: python

    import tltorch
    import torch

Let’s create a batch of 4 data points of size 16 each:

.. code-block:: python

    data = torch.randn((4, 16), dtype=torch.float32)

Now, imagine you already have a linear layer:

.. code-block:: python

    linear = torch.nn.Linear(in_features=16, 10)

You can easily compress it into a tensorized linear layer: here we specify the shape to which to tensorize the weights,
and use `rank=0.5`, which means automatically determine the rank so that the factorization uses approximately half the
number of parameters.

.. code-block:: python

    fact_linear = tltorch.FactorizedLinear.from_linear(linear, auto_tensorize=False,
                        in_tensorized_features=(4, 4), out_tensorized_features=(2, 5), rank=0.5)


The tensorized weights will have the following shape: 

.. parsed-literal::

    torch.Size([4, 4, 2, 5])


Note that you can also let TensorLy-Torch automatically determine the tensorization shape. In this case we just instruct it to 
find ``in_tensorized_features`` and ``out_tensorized_features`` to have length `2`:

.. code-block:: python

    fact_linear = tltorch.FactorizedLinear.from_linear(linear, auto_tensorize=True, n_tensorized_modes=2, rank=0.5)


You can also create tensorized layers from scratch:

.. code-block:: python

    fact_linear = tltorch.FactorizedLinear(in_tensorized_features=(4, 4), 
                                           out_tensorized_features=(2, 5), 
                                           factorization='tucker', rank=0.5)

Finally, during the forward pass, you can reconstruct the full weights (``implementation='reconstructed'``) and perform a regular linear layer forward pass. 
ALternatively, you can let TensorLy-Torch automatically direction contract the input tensor with the *factors of the decomposition*  (``implementation='factorized'``),
 which can be faster, particularly if you have a very small rank, e.g. very small factorization factors. 