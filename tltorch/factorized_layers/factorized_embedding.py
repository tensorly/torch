from itertools import cycle, islice
import torch
import numpy as np
from torch import nn
from torch.nn.functional import embedding
from scipy.stats import entropy
from sympy.utilities.iterables import multiset_partitions
from sympy.ntheory import factorint
from ..factorized_tensors import TensorizedTensor
# Author: Cole Hawkins, with reshaping code taken from https://github.com/KhrulkovV/tt-pytorch
# License: BSD 3 clause

MODES = ['ascending', 'descending', 'mixed']
CRITERIONS = ['entropy', 'var']

def _to_list(p):
    res = []
    for k, v in p.items():
        res += [
            k,
        ] * v
    return res


def _roundup(n, k):
    return int(np.ceil(n / 10**k)) * 10**k


def _roundrobin(*iterables):
    "roundrobin('ABC', 'D', 'EF') --> A D E B F C"
    # Recipe credited to George Sakkis
    pending = len(iterables)
    nexts = cycle(iter(it).__next__ for it in iterables)
    while pending:
        try:
            for next in nexts:
                yield next()
        except StopIteration:
            pending -= 1
            nexts = cycle(islice(nexts, pending))


def _get_all_factors(n, d=3, mode='ascending'):
    p = _factorint2(n)
    if len(p) < d:
        p = p + [
            1,
        ] * (d - len(p))

    if mode == 'ascending':

        def prepr(x):
            return tuple(sorted([np.prod(_) for _ in x]))
    elif mode == 'descending':

        def prepr(x):
            return tuple(sorted([np.prod(_) for _ in x], reverse=True))

    elif mode == 'mixed':

        def prepr(x):
            x = sorted(np.prod(_) for _ in x)
            N = len(x)
            xf, xl = x[:N // 2], x[N // 2:]
            return tuple(_roundrobin(xf, xl))

    else:
        raise ValueError(
            'Wrong mode specified, only {} are available'.format(MODES))

    raw_factors = multiset_partitions(p, d)
    clean_factors = [prepr(f) for f in raw_factors]
    clean_factors = list(set(clean_factors))
    return clean_factors


def _factorint2(p):
    return _to_list(factorint(p))


def auto_shape(n, d=3, criterion='entropy', mode='ascending'):
    factors = _get_all_factors(n, d=d, mode=mode)
    if criterion == 'entropy':
        weights = [entropy(f) for f in factors]
    elif criterion == 'var':
        weights = [-np.var(f) for f in factors]
    else:
        raise ValueError(
            'Wrong criterion specified, only {} are available'.format(
                CRITERIONS))

    i = np.argmax(weights)
    return list(factors[i])


def suggest_shape(n, d=3, criterion='entropy', mode='ascending'):
    """Given n, round up n to an easy number to factorize and return that factorization
    """
    weights = []
    for i in range(len(str(n))):

        n_i = _roundup(n, i)
        if criterion == 'entropy':
            weights.append(
                entropy(auto_shape(n_i, d=d, mode=mode, criterion=criterion)))
        elif criterion == 'var':
            weights.append(
                -np.var(auto_shape(n_i, d=d, mode=mode, criterion=criterion)))
        else:
            raise ValueError(
                'Wrong criterion specified, only {} are available'.format(
                    CRITERIONS))

    i = np.argmax(weights)
    factors = auto_shape(int(_roundup(n, i)),
                         d=d,
                         mode=mode,
                         criterion=criterion)
    return factors


class FactorizedEmbedding(nn.Module):
    """
    Tensorized Embedding Layers For Efficient Model Compression

    Tensorized drop-in replacement for torch.nn.Embedding 

    Parameters
    ----------
    num_embeddings : int, number of entries in the lookup table
    embedding_dim : int, number of dimensions per entry
    auto_reshape : bool, whether to use automatic reshaping for the embedding dimensions
    d : int or int tuple, number of reshape dimensions for both embedding table dimension
    tensorized_num_embeddings : int tuple, tensorized shape of the first embedding table dimension
    tensorized_embedding_dim : int tuple, tensorized shape of the second embedding table dimension
    rank : int tuple or str, tensor rank
    """
    def __init__(self,
                 num_embeddings,
                 embedding_dim,
                 auto_reshape=True,
                 d=(3, 3),
                 tensorized_num_embeddings=None,
                 tensorized_embedding_dim=None,
                 factorization='blocktt',
                 rank=8,
                 device=None,
                 dtype=None):
        super().__init__()

        if auto_reshape:
            try:
                assert tensorized_num_embeddings is None and tensorized_embedding_dim is None
            except:
                raise ValueError(
                    "Automated factorization enabled but tensor dimensions provided"
                )

            #if user provides an int, expand to tuple and assume each embedding dimension gets reshaped to the same number of dimensions
            if type(d) == int:
                d = (d, d)

            tensorized_num_embeddings = suggest_shape(num_embeddings, d[0])
            tensorized_embedding_dim = suggest_shape(embedding_dim, d[1])
            num_embeddings = np.prod(tensorized_num_embeddings)
            embedding_dim = np.prod(tensorized_embedding_dim)
        else:
            try:
                assert np.prod(
                    tensorized_num_embeddings) == num_embeddings and np.prod(
                        tensorized_embedding_dim) == embedding_dim
            except:
                raise ValueError(
                    "Must provide appropriate reshaping dimensions")

        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.tensor_shape = (tensorized_num_embeddings,
                             tensorized_embedding_dim)
        self.weight_shape = (self.num_embeddings, self.embedding_dim)

        self.factorization = factorization

        self.weight = TensorizedTensor.new(self.tensor_shape,
                                           rank=rank,
                                           factorization=self.factorization,
                                           device=device,
                                           dtype=dtype)
        self.reset_parameters()

        self.rank = self.weight.rank

    def reset_parameters(self):
        #Parameter initialization from Yin et al.
        #TT-Rec: Tensor Train Compression for Deep Learning Recommendation Model Embeddings
        target_stddev = 1 / np.sqrt(3 * self.num_embeddings)
        with torch.no_grad():
            self.weight.normal_(0, target_stddev)

    def forward(self, input, offsets=None):

        return embedding(input, self.weight, offsets)
        #get item from tensor here
