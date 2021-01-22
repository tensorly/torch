:no-toc:
:no-localtoc:
:no-pagination:

.. TensorLy-Torch documentation

.. only:: html

   .. raw:: html

      <div class="container content">
      <br/><br/>

.. image:: _static/logos/tensorly-torch-logo.png
   :align: center
   :width: 500


.. only:: html

   .. raw:: html 
   
      <div class="has-text-centered">
         <h3> Tensor Batteries Included </h3>
      </div>
      <br/><br/>


**TensorLy-Torch** is a PyTorch only library that builds on top of `TensorLy <http://tensorly.org/dev>`_ and provides out-of-the-box tensor layers.
It comes with all batteries included and tries to make it as easy as possible to use tensor methods within your deep networks. 

- **Leverage structure in your data**: with tensor layers, you can easily leverage the structure in your data, through :mod:`TRL <tltorch._trl>`, :mod:`TCL <tltorch._tcl>`, :mod:`Factorized convolutions <tltorch.factorized_conv>` and more!
- **Built-in tensor layers**: all you have to do is import tensorly torch and include the layers we provide directly within your PyTorch models!
- **Initialization**: initializing tensor decompositions can be tricky. We take care of it all, whether you want to initialize randomly using our :mod:`tltorch.init` module or from a pretrained layer.
- **Tensor hooks**: you can easily augment your architectures with our built-in hooks. Robustify your network with :mod:`Tensor Dropout <tltorch._tensor_dropout>`. Automatically select the rank end-to-end with L1 Regularization!
- **All the methods available**: we are always adding more methods to make it easy to compare between the performance of various deep tensor based methods!

Deep Tensorized Learning
========================

Tensor methods generalize matrix algebraic operations to higher-orders. Deep neural networks typically map between higher-order tensors. 
In fact, it is the ability of deep convolutional neural networks to preserve and leverage local structure that, along with large datasets and efficient hardware, made the current levels of performance possible.
Tensor methods allow to further leverage and preserve that structure, for individual layers or whole networks. 

.. image:: _static/tensorly-torch-pyramid.png
   :align: center
   :width: 800

TensorLy is a Python library that aims at making tensor learning simple and accessible.
It provides a high-level API for tensor methods, including core tensor operations, tensor decomposition and regression. 
It has a flexible backend that allows running operations seamlessly using NumPy, PyTorch, TensorFlow, JAX, MXNet and CuPy.
 
**TensorLy-Torch** is a PyTorch only library that builds on top of TensorLy and provides out-of-the-box tensor layers.

Improve your neural networks with tensor methods
------------------------------------------------

Tensor methods generalize matrix algebraic operations to higher-orders. Deep neural networks typically map between higher-order tensors. 
In fact, it is the ability of deep convolutional neural networks to preserve and leverage local structure that, along with large datasets and efficient hardware, made the current levels of performance possible.
Tensor methods allow to further leverage and preserve that structure, for individual layers or whole networks. 

.. image:: _static/deep_tensor_nets_pros_circle.png
   :align: center
   :width: 350


In TensorLy-Torch, we provide convenient layers that do all the heavy lifting for you 
and provide the benefits tensor based layers wrapped in a nice, well documented and tested API.

For instance, convolution layers of any order (2D, 3D or more), can be efficiently parametrized
using tensor decomposition. Using a CP decomposition results in a separable convolution
and you can replace your original convolution with a series of small efficient ones: 

.. image:: _static/cp-conv.png 
   :width: 500
   :align: center

These can be easily perform with the :ref:`factorized_conv_ref` module in TensorLy-Torch.
We also have Tucker convolutions and new tensor-train convolutions!
We also implement various other methods such as tensor regression and contraction layers, 
tensorized linear layers, tensor dropout and more!


.. toctree::
   :maxdepth: 1
   :hidden:

   install
   user_guide/index
   modules/api
   dev_guide/index
   about
   /tensor_regression_layers

.. only:: html

   .. raw:: html

      <br/> <br/>
      <br/>

      <div class="container has-text-centered">
      <a class="button is-large is-dark is-primary" href="install.html">
         Start Tensorizing!
      </a>
      </div>
      
      </div>
