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

You can easily compress it into a tensorized linear layer:

.. code-block:: python

    fact_linear = tltorch.CPLinear.from_linear(linear, tensorized_shape=(4, 4, 2, 5), rank=0.5)


.. parsed-literal::

    torch.Size([4, 4, 2, 5])


You can also create tensorized layers from scratch:

.. code-block:: python

    fact_linear = tltorch.CPLinear(in_features=16, out_features=10,
                                   tensorized_shape=(4, 4, 2, 5), rank=0.5)
