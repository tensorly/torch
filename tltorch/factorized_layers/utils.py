import numpy as np
from itertools import cycle, islice
from scipy.stats import entropy
from sympy.utilities.iterables import multiset_partitions
from sympy.ntheory import factorint

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

