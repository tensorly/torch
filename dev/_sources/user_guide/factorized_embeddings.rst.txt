Factorized embedding layers
===========================

In TensorLy-Torch, we also provide out-of-the-box tensorized embedding layers. 

Just as for the case of factorized linear, you can either create a factorized embedding from scratch, here automatically determine the 
input and output tensorized shapes, to have 3 dimensions each:

.. code-block:: python

    import tltorch
    import torch

    from_embedding = tltorch.FactorizedEmbedding(num_embeddings, embedding_dim, auto_reshape=True, d=3, rank=0.4)


Or, you can create it by decomposing an existing embedding layer:

    from_embedding = tltorch.FactorizedEmbedding.from_embedding(embedding_layer, auto_reshape=True,
                factorization='blocktt', n_tensorized_modes=3, rank=0.4)