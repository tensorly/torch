=============
API reference
=============

:mod:`tltorch`: Tensorized Deep Neural Networks

.. automodule:: tltorch
    :no-members:
    :no-inherited-members:

.. _factorized_tensor_ref:

Factorized Tensors
==================

TensorLy-Torch builds on top of TensorLy and provides out of the box PyTorch layers for tensor based operations.
The core of this is the concept of factorized tensors, which factorize our layers, instead of regular, dense PyTorch tensors.

You can create any factorized tensor through the main class using:

.. autosummary::
    :toctree: generated
    :template: class.rst

    FactorizedTensor

You can create a tensor of any form using ``FactorizedTensor.new(shape, rank, factorization)``, where factorization can be `Dense`, `CP`, `Tucker` or `TT`.
Note that if you use ``factorization = 'dense'`` you are just creating a regular, unfactorized tensor. 
This allows to manipulate any tensor, factorized or not, with a simple, unified interface.

Alternatively, you can also directly create a specific subclass:

.. autosummary::
    :toctree: generated
    :template: class.rst

    DenseTensor
    CPTensor
    TuckerTensor
    TTTensor

.. _factorized_matrix_ref:

Tensorized Matrices
===================

In TensorLy-Torch , you can also represent matrices in *tensorized* form, as low-rank tensors.
 Just as for factorized tensor, you can create a tensorized matrix through the main class using:

.. autosummary::
    :toctree: generated
    :template: class.rst

    TensorizedTensor

You can create a tensor of any form using ``TensorizedTensor.new(tensorized_shape, rank, factorization)``, where factorization can be `Dense`, `CP`, `Tucker` or `BlockTT`.

You can also explicitly create the type of tensor you want using the following classes:

.. autosummary::
    :toctree: generated
    :template: class.rst

    DenseTensorized
    TensorizedTensor
    CPTensorized
    BlockTT

.. _complex_ref:

Complex Tensors
===============

In theory, you can simply specify ``dtype=torch.cfloat`` in the creation of any of the tensors of tensorized matrices above, to automatically create a complex valued tensor.
However, in practice, there are many issues in complex support. Distributed Data Parallelism in particular, is not supported.

In TensorLy-Torch, we propose a convenient and transparent way around this: simply use ``ComplexTensor`` instead. 
This will store the factors of the decomposition in real form (by explicitly storing the real and imaginary parts) 
but will transparently return you a complex valued tensor or reconstruction.

.. autosummary::
    :toctree: generated
    :template: class.rst

    ComplexDenseTensor
    ComplexCPTensor
    ComplexTuckerTensor
    ComplexTTTensor


    ComplexDenseTensorized
    ComplexTuckerTensorized
    ComplexCPTensorized
    ComplexBlockTT


You can also transparently instanciate any of these using directly the main classes, ``TensorizedTensor`` or ``FactorizedTensor`` and specifying 
``factorization="ComplexCP"`` or in general ``ComplexFactorization`` with `Factorization` any of the supported decompositions.


.. _init_ref:

Initialization
==============

.. automodule:: tltorch.factorized_tensors
    :no-members:
    :no-inherited-members:

Initialization is particularly important in the context of deep learning. 
We provide convenient functions to directly initialize factorized tensor (i.e. their factors)
such that their reconstruction follows approximately a centered Gaussian distribution. 

.. currentmodule:: tltorch.factorized_tensors.init

.. autosummary::
    :toctree: generated
    :template: function.rst

    tensor_init
    cp_init
    tucker_init
    tt_init
    block_tt_init

.. _trl_ref:

Tensor Regression Layers
========================

.. automodule:: tltorch.factorized_layers
    :no-members:
    :no-inherited-members:

.. currentmodule:: tltorch.factorized_layers

.. autosummary::
    :toctree: generated
    :template: class.rst

    TRL

.. _tcl_ref:

Tensor Contraction Layers
=========================

.. autosummary::
    :toctree: generated
    :template: class.rst

    TCL

.. _factorized_linear_ref:

Factorized Linear Layers
========================

.. autosummary::
    :toctree: generated
    :template: class.rst

    FactorizedLinear

.. _factorized_conv_ref:

Factorized Convolutions
=======================

General N-Dimensional convolutions in Factorized forms

.. autosummary::
    :toctree: generated
    :template: class.rst

    FactorizedConv

.. _tensor_dropout_ref:

Factorized Embeddings
=====================

A drop-in replacement for PyTorch's embeddings but using an efficient tensor parametrization that never reconstructs the full table. 

.. autosummary::
    :toctree: generated
    :template: class.rst

    FactorizedEmbedding

.. _tensor_dropout_ref:

Tensor Dropout
==============

.. currentmodule:: tltorch.tensor_hooks

.. automodule:: tltorch.tensor_hooks
    :no-members:
    :no-inherited-members:

These functions allow you to easily add or remove tensor dropout from tensor layers.


.. autosummary::
    :toctree: generated
    :template: function.rst

    tensor_dropout
    remove_tensor_dropout


You can also use the class API below but unless you have a particular use for the classes, you should use the convenient functions provided instead.

.. autosummary::
    :toctree: generated
    :template: class.rst

    TensorDropout

.. _tensor_lasso_ref:

L1 Regularization
=================

L1 Regularization on tensor modules. 

.. currentmodule:: tltorch.tensor_hooks

.. autosummary::
    :toctree: generated
    :template: function.rst

    tensor_lasso
    remove_tensor_lasso

Utilities
=========

Utility functions

.. currentmodule:: tltorch.utils

.. autosummary::
    :toctree: generated
    :template: function.rst

    get_tensorized_shape