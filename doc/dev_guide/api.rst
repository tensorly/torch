Style and API
=============

In TensorLy-Torch (and more generally in the TensorLy project),
we try to maintain a simple and consistent API.

Here are some elements to consider.

TensorModules
-------------
:class:`tltorch.TensorModule` is a base class we provide 
for PyTorch modules parametrized by a tensor decomposition. 

They must implement `get_decomposition()`
 which returns the decomposition parametrizing the tensor layer.
In the forward pass, the decomposition should be accessed
 through :meth:`tltorch.TensorModule._process_decomposition`,
  not directly as `module.factors` for instance.

This is to make sure all the decomposition hooks are applied first
(i.e. :mod:`tensor dropout <tltorch._tensor_dropout>`).

Initialization
--------------
Modules should follow the following API for initialization: for initializating an instance,
 use `init_from_random`, `init_from_decompostion` and `init_from_tensor`. 

Class method should be used to create an instance 
and initialize it from existing weights or module (e.g. `Class.from_tensor`, etc).

