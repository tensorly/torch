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

You can create any factorized tensor through the main class, or directly create a specific subclass:

.. autosummary::
    :toctree: generated
    :template: class.rst

    FactorizedTensor
    CPTensor
    TuckerTensor
    TTTensor

.. _factorized_matrix_ref:

Tensorized Matrices
===================

In TensorLy-Torch , you can also represent matrices in *tensorized* form, as low-rank tensors . 

.. autosummary::
    :toctree: generated
    :template: class.rst

    TensorizedTensor
    CPTensorized
    BlockTT

.. _init_ref:

Initialization
==============

.. automodule:: tltorch.init
    :no-members:
    :no-inherited-members:

.. currentmodule:: tltorch.init

.. autosummary::
    :toctree: generated
    :template: function.rst

    tensor_init
    cp_init
    tucker_init
    tt_init


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
