import pytest
import torch
from torch import nn
from ..factorized_embedding import FactorizedEmbedding

import tensorly as tl
tl.set_backend('pytorch')
from tensorly import testing

@pytest.mark.parametrize('factorization', ['CP', 'Tucker', 'BlockTT'])
@pytest.mark.parametrize('dims', [(997,16), (33,7)])
def test_FactorizedEmbedding(factorization,dims):

    
    
    NUM_EMBEDDINGS,EMBEDDING_DIM=dims
    BATCH_SIZE = 3

    #create factorized embedding
    factorized_embedding = FactorizedEmbedding(NUM_EMBEDDINGS,EMBEDDING_DIM,factorization=factorization)

    #make test embedding of same shape and same weight
    test_embedding = torch.nn.Embedding(factorized_embedding.weight.shape[0],EMBEDDING_DIM) 
    test_embedding.weight.data.copy_(factorized_embedding.weight.to_matrix().detach())
    
    #create batch for lookup
    batch = torch.randperm(NUM_EMBEDDINGS)[:BATCH_SIZE].view(-1,1)

    normal_embed = test_embedding(batch)
    factorized_embed = factorized_embedding(batch)

    testing.assert_array_almost_equal(normal_embed,factorized_embed)

