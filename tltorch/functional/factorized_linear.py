import torch
import tensorly as tl
from collections import Counter
from tensorly.tt_tensor import TTTensor

tl.set_backend('pytorch')

# Author: Taylor Lee Patti <taylorpatti@g.harvard.edu>

def tt_factorized_linear(x, weights):
    ncores = len(x)
    contr_layer = []
    for i in range(ncores):
        dimW, dimX = weights[i].shape, x[i].shape
        contr = tl.einsum('abc,debf->adecf', x[i], weights[i])
        contr_layer.append(tl.reshape(contr, (dimW[0]*dimX[0], dimW[1], dimW[3]*dimX[2])))
    return TTTensor(contr_layer)
