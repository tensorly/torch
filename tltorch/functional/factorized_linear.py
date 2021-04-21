import torch
import tensorly as tl
import opt_einsum as oe
from collections import Counter

tl.set_backend('pytorch')

# Author: Taylor Lee Patti <taylorpatti@g.harvard.edu>

def tt_factorized_linear(x, weights):
    ncores = len(x)
    eq = _contraction_eq(ncores)
    return oe.contract(eq, *weights, *x)

def _contraction_eq(ncores):
    start = 1
    x_idx = []
    for i in range(ncores):
            idx = [start+2*i, start+1+2*i, start+2+2*i]
            x_idx.append(''.join(oe.parser.get_symbol(j) for j in idx))
    start2 = start+2+2*i
    weights_idx = []
    for i in range(ncores):
        if i==0:
            idx = [start, start2+1, start+1, start2+2]
        elif i==ncores-1:
            idx = [start2+2*i, start2+2*i+1, start+1+2*i, start2]
        else:
            idx = [start2+2*i, start2+2*i+1, start+1+2*i, start2+2*i+2]
        weights_idx.append(''.join(oe.parser.get_symbol(j) for j in idx))
    out_idx = ''.join(weights_idx)+''.join(x_idx)
    counts = Counter(out_idx)
    out_idx = ''.join(ind for ind, count in counts.items() if count == 1)

    return ','.join(i for i in weights_idx) + ',' + ','.join(i for i in x_idx) + '->' + out_idx
