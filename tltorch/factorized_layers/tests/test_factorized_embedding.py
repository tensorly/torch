import pytest
import torch
from torch import nn
from ..factorized_embedding import FactorizedEmbedding

import tensorly as tl
tl.set_backend('pytorch')
from tensorly import testing

#@pytest.mark.parametrize('factorization', ['BlockTT'])
@pytest.mark.parametrize('factorization', ['CP','Tucker', 'BlockTT'])
@pytest.mark.parametrize('dims', [(256,16), (1000,32)])
def test_FactorizedEmbedding(factorization,dims):

    
    
    NUM_EMBEDDINGS,EMBEDDING_DIM=dims
    BATCH_SIZE = 3

    #create factorized embedding
    factorized_embedding = FactorizedEmbedding(NUM_EMBEDDINGS,EMBEDDING_DIM,factorization=factorization)

    #make test embedding of same shape and same weight
    test_embedding = torch.nn.Embedding(factorized_embedding.weight.shape[0],factorized_embedding.weight.shape[1]) 
    test_embedding.weight.data.copy_(factorized_embedding.weight.to_matrix().detach())
    
    #create batch and test using all entries (shuffled since entries may not be sorted)
    batch = torch.randperm(NUM_EMBEDDINGS)#.view(-1,1)
    normal_embed = test_embedding(batch)
    factorized_embed = factorized_embedding(batch)
    testing.assert_array_almost_equal(normal_embed,factorized_embed,decimal=2)
    
    #split batch into tensor with first dimension 3
    batch = torch.randperm(NUM_EMBEDDINGS)
    split_size = NUM_EMBEDDINGS//5
   
    split_batch = [batch[:1*split_size],batch[1*split_size:2*split_size],batch[3*split_size:4*split_size]]

    split_batch = torch.stack(split_batch,0)

    normal_embed = test_embedding(split_batch)
    factorized_embed = factorized_embedding(split_batch)
    testing.assert_array_almost_equal(normal_embed,factorized_embed,decimal=2)

    #BlockTT has no init_from_matrix, so skip that test
    if factorization=='BlockTT':
        return

    del factorized_embedding

    #init from test layer which is low rank
    factorized_embedding = FactorizedEmbedding.from_embedding(test_embedding,factorization=factorization,rank=8)
    
    #test using same batch as before, only test that shapes match
    normal_embed = test_embedding(batch)
    factorized_embed = factorized_embedding(batch)
    testing.assert_array_almost_equal(normal_embed.shape,factorized_embed.shape,decimal=2)
