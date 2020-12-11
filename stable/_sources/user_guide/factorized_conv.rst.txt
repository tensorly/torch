Factorized Convolutional Layers
===============================

It is possible to apply low-rank tensor factorization to convolution
kernels to compress the network and reduce the number of parameters.

In TensorLy-Torch, you can easily try factorized convolutions: first, let’s
import the library:

.. code:: python

    import tltorch
    import torch

Let’s now create some random data to try our modules: we can choose the
size of the convolutions.

.. code:: python

    device='cpu'
    input_channels = 16
    output_channels = 32
    kernel_size = 3
    batch_size = 2
    size = 24
    order = 2
    
    input_shape = (batch_size, input_channels) + (size, )*order
    kernel_shape = (output_channels, input_channels) + (kernel_size, )*order

We can create some random input data:

.. code:: python

    data = torch.randn(input_shape, dtype=torch.float32, device=device)

Creating Factorized Convolutions
--------------------------------

From Random
~~~~~~~~~~~

In PyTorch, you would create a convolution as follows:

.. code:: python

   conv = torch.nn.Conv2d(input_channels, output_channels, kernel_size)

In TensorLy Torch, it is exactly the same except that factorized
convolutions are by default of any order: either you specify the kernel
size or your specify the order

.. code:: python

   conv = tltorch.TuckerConv(input_channels, output_channels, kernel_size, order=2, rank='same')

.. code:: python

    conv = torch.nn.Conv2d(input_channels, output_channels, kernel_size=3)

In TensorLy-Torch, factorized convolutions can be of any order, so you
have to specify the order at creation (in Pytorch, you specify it
through the class name, e.g. Conv2d or Conv3d):

.. code:: python

    fact_conv = tltorch.CPConv(input_channels, output_channels, kernel_size=3, order=2, rank='same')


Or, you can specify the order directly by passing a tuple as kernel_size
(in which case, ``order = len(kernel_size)`` is used).

.. code:: python

    fact_conv = tltorch.CPConv(input_channels, output_channels, kernel_size=(3, 3), rank='same')

From an existing Convolution
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

You can create a Factorized convolution from an existing (PyTorch)
convolution:

.. code:: python

    fact_conv = tltorch.CPConv.from_conv(conv, rank=0.5, decompose_weights=True)

Efficient Convolutional Blocks
------------------------------

If you compress a convolutional kernel, you can get efficient
convolutional blocks by applying tensor factorization. For instance, if
you apply CP decomposition, you can get a MobileNet-v2 block:

.. code:: python

    fact_conv = tltorch.CPConv.from_conv(conv, rank=0.5, implementation='mobilenet')


Similarly, if you apply Tucker decomposition, you can get a ResNet
BottleNeck block:

.. code:: python

    fact_conv = tltorch.TuckerConv.from_conv(conv, rank=0.5, implementation='factorized')
