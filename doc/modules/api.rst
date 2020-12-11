=============
API reference
=============

:mod:`tltorch`: Tensorized Deep Neural Networks

.. automodule:: tltorch
    :no-members:
    :no-inherited-members:

.. _trl_ref:

Tensor Regression Layers
========================

.. autosummary::
    :toctree: generated
    :template: class.rst

    TuckerTRL
    CPTRL
    TensorTrainTRL

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

    TTLinear
    TTMLinear
    CPLinear
    TuckerLinear

.. _factorized_conv_ref:

Factorized Convolutions
=======================

:mod:`tltorch.factorized_conv`: General N-Dimensional convolutions in Factorized forms

.. automodule:: tltorch.factorized_conv
    :no-members:
    :no-inherited-members:

.. currentmodule:: tltorch.factorized_conv

.. autosummary::
    :toctree: generated
    :template: class.rst

    TuckerConv
    CPConv
    TTConv


.. _tensor_dropout_ref:

Tensor Dropout
==============

.. currentmodule:: tltorch._tensor_dropout

.. automodule:: tltorch._tensor_dropout
    :no-members:
    :no-inherited-members:

Classes
-------

Unless you have a particular use for the classes, you should use the convenient functions provided instead.

.. autosummary::
    :toctree: generated
    :template: class.rst

    TuckerDropout
    CPDropout
    TTDropout


Functions
---------

Convenience functions to easily add or remove tensor dropout from tensor layers.


.. autosummary::
    :toctree: generated

    tucker_dropout
    cp_dropout
    tt_dropout
    remove_tucker_dropout
    remove_cp_dropout
    remove_tt_dropout


.. _tensor_lasso_ref:

L1 Regularization
=================

L1 Regularization on tensor modules. 

.. currentmodule:: tltorch._tensor_lasso


.. autosummary::
    :toctree: generated
    :template: class.rst

    TuckerL1Regularizer
    CPL1Regularizer
    TTL1Regularizer

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

    cp_init
    tucker_init
    tt_init


.. _internal_ref:

Internal
========

.. currentmodule:: tltorch.base

.. autosummary::
    :toctree: generated
    :template: class

    TensorModule