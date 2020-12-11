Tensor Hooks
============

TensorLy-Torch also makes it very easy to manipulate the tensor decomposition parametrizing a tensor module.

Tensor dropout
--------------
For instance, you can apply very easily tensor dropout: let's first create a simple TRL layer.

.. code:: python

   import tltorch
   trl = tltorch.TuckerTRL((10, 10), (10, ), rank='same')

To add tensor dropout, simply apply the helper function:

.. code:: python

   trl = tltorch.tucker_dropout(trl, p=0.5)


Similarly, to remove tensor dropout:

.. code:: python

   tltorch.remove_tucker_dropout(trl)


Lasso rank regularization
-------------------------
Rank selection is a hard problem. One way to choose the rank while training is to apply 
an l1 penalty (Lasso) on the rank.

This was used previously for CP decomposition, and we extended it in TensorLy-Torch to Tucker and Tensor-Train,
by introducing new weights in the decomposition. 

To use is, you can define a regularizer object that will take care of everything. 
Using our previously defined TRL:

.. code:: python

   l1_reg = tltorch.TuckerL1Regularizer(penalty=0.01)
   l1_reg.apply(trl)
   x = trl(x)
   loss = my_loss(x) + l1_reg.loss
   l1_reg.res

After each iteration, don't forget to reset the loss so you don't keep accumulating:

.. code:: python

   l1_reg.reset()

Initializing tensor decomposition
---------------------------------

Another issue is that of initializing tensor decompositions: 
if you simply initialize randomly each component without care, 
the reconstructed (full) tensor can have arbitrarily large or small values
potentially leading to gradient vanishing or exploding during training.

In TensorLy-Torch, we provide a module for initialization that will 
properly initialize the factors of the decomposition 
so that the reconstruction has zero mean and the specified standard deviation!

For instance, for a CP tensor ``(weights, factors)``:

.. code:: python

   tltorch.init.cp_init(weights, factors)


