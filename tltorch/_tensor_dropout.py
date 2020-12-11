"""Tensor Dropout for TensorModules"""

# Author: Jean Kossaifi
# License: BSD 3 clause

import tensorly as tl
tl.set_backend('pytorch')
import torch

class TuckerDropout():
    """Decomposition Hook for Tensor Dropout on Tucker tensors
    
    Parameters
    ----------
    proba : float, probability of dropout
    min_dim : int
        Minimum dimension size for which to apply dropout.
        For instance, if a tensor if of shape (32, 32, 3, 3) and min_dim = 4
        then dropout will *not* be applied to the last two modes.
    """
    def __init__(self, proba, min_dim=1):
        self.proba = proba
        self.min_dim = min_dim
        self.fun = self._apply_tensor_dropout

    def __call__(self, module, tucker_tensor):
        return self.fun(tucker_tensor)
        #return self._apply_tensor_dropout_numpy(tucker_tensor)
            
    def _apply_tensor_dropout(self, tucker_tensor):
        core, factors = tucker_tensor
        tucker_rank = core.shape

        sampled_indices = []
        for rank in tucker_rank:
            idx = tl.arange(rank, device=core.device, dtype=torch.int64)
            if rank > self.min_dim:
                idx = idx[torch.bernoulli(torch.ones(rank, device=core.device)*(1 - self.proba),
                                      out=torch.empty(rank, device=core.device, dtype=torch.bool))]
                if len(idx) == 0:
                    idx = torch.randint(0, rank, size=(1, ), device=core.device, dtype=torch.int64)

            sampled_indices.append(idx)
        
        core = core[torch.meshgrid(*sampled_indices)]
        factors = [factor[:, idx]  for (factor, idx) in zip(factors, sampled_indices)]

        return core, factors

    @staticmethod
    def apply(module, proba, min_dim=1):
        dropout = TuckerDropout(proba, min_dim=min_dim)
        handle = module.register_decomposition_forward_pre_hook(dropout)
        return handle

 
class CPDropout():
    """Decomposition Hook for Tensor Dropout on Tucker tensors
    
    Parameters
    ----------
    p : float, probability of dropout
    min_dim : int
        Minimum dimension size for which to apply dropout.
        For instance, if a tensor if of shape (32, 32, 3, 3) and min_dim = 4
        then dropout will *not* be applied to the last two modes.
    """
    def __init__(self, proba, min_dim=1):
        self.proba = proba
        self.min_dim = min_dim

    def __call__(self, module, cp_tensor):
        return self._apply_tensor_dropout(cp_tensor)
            
    def _apply_tensor_dropout(self, cp_tensor):
        weights, factors = cp_tensor
        rank = factors[0].shape[1]
        device = factors[0].device
        
        if rank > self.min_dim:
            sampled_indices = tl.arange(rank, device=device, dtype=torch.int64)
            sampled_indices = sampled_indices[torch.bernoulli(torch.ones(rank, device=device)*(1 - self.proba),
                                  out=torch.empty(rank, device=device, dtype=torch.bool))]
            if len(sampled_indices) == 0:
                sampled_indices = torch.randint(0, rank, size=(1, ), device=device, dtype=torch.int64)

            factors = [factor[:, sampled_indices] for factor in factors]
            weights = weights[sampled_indices]

        return weights, factors

    @staticmethod
    def apply(module, proba, min_dim=1):
        dropout = CPDropout(proba, min_dim=min_dim)
        handle = module.register_decomposition_forward_pre_hook(dropout)
        return handle
    
class TTDropout():
    """Decomposition Hook for Tensor Dropout on Tucker tensors
    
    Parameters
    ----------
    p : float, probability of dropout
    min_dim : int
        Minimum dimension size for which to apply dropout.
        For instance, if a tensor if of shape (32, 32, 3, 3) and min_dim = 4
        then dropout will *not* be applied to the last two modes.
    """
    def __init__(self, proba, min_dim=1):
        self.proba = proba
        self.min_dim = min_dim

    def __call__(self, module, tt_tensor):
        return self._apply_tensor_dropout(tt_tensor)
            
    def _apply_tensor_dropout(self, tt_tensor):
        factors = tt_tensor
        order = len(factors)
        tt_rank = [f.shape[0] for f in factors[1:]]
        device = factors[0].device

        sampled_indices = []
        for i, rank in enumerate(tt_rank):
            if rank > self.min_dim:
                idx = tl.arange(rank, device=device, dtype=torch.int64)
                idx = idx[torch.bernoulli(torch.ones(rank, device=device)*(1 - self.proba),
                                      out=torch.empty(rank, device=device, dtype=torch.bool))]
                if len(idx) == 0:
                    idx = torch.randint(0, rank, size=(1, ), device=device, dtype=torch.int64)
            else:
                idx = tl.arange(rank, **tl.context(factors[0])).tolist()

            sampled_indices.append(idx)

        sampled_factors = []
        for i, f in enumerate(factors):
            if i == 0:
                sampled_factors.append(f[..., sampled_indices[i]])
            elif i == (order - 1):
                sampled_factors.append(f[sampled_indices[i-1], ...])
            else:
                sampled_factors.append(f[sampled_indices[i-1], ...][..., sampled_indices[i]])

        return sampled_factors

    @staticmethod
    def apply(module, proba, min_dim=1):
        dropout = TTDropout(proba, min_dim=min_dim)
        handle = module.register_decomposition_forward_pre_hook(dropout)
        return handle


def tucker_dropout(module, p):
    """Tucker Dropout

    Parameters
    ----------
    module : tltorch.TensorModule
        the tensor module parametrized by the tensor decomposition to which to apply tensor dropout
    p : float
        dropout probability
        if 0, no dropout is applied
        if 1, all the components but 1 are dropped in the latent space

    Returns
    -------
    TensorModule
        the module to which tensor dropout has been attached

    Examples
    --------
    >>> trl = tltorch.TuckerTRL((10, 10), (10, ), rank='same')
    >>> trl = tucker_dropout(trl, p=0.5)
    >>> remove_tucker_dropout(trl)
    """
    TuckerDropout.apply(module, p, min_dim=1)
    return module

def remove_tucker_dropout(module):
    """Removes the tensor dropout from a TensorModule 

    Parameters
    ----------
    module : tltorch.TensorModule
        the tensor module parametrized by the tensor decomposition to which to apply tensor dropout
    
    Examples
    --------
    >>> trl = tltorch.TuckerTRL((10, 10), (10, ), rank='same')
    >>> trl = tucker_dropout(trl, p=0.5)
    >>> remove_tucker_dropout(trl)
    """
    for key, hook in module._decomposition_forward_pre_hooks.items():
        if isinstance(hook, TuckerDropout):
            del module._decomposition_forward_pre_hooks[key]
            break
            
def cp_dropout(module, p):
    """CP Dropout

    Parameters
    ----------
    module : tltorch.TensorModule
        the tensor module parametrized by the tensor decomposition to which to apply tensor dropout
    p : float
        dropout probability
        if 0, no dropout is applied
        if 1, all the components but 1 are dropped in the latent space

    Returns
    -------
    TensorModule
        the module to which tensor dropout has been attached

    Examples
    --------
    >>> trl = tltorch.CPTRL((10, 10), (10, ), rank='same')
    >>> trl = cp_dropout(trl, p=0.5)
    >>> remove_cp_dropout(trl)  
    """
    CPDropout.apply(module, p, min_dim=1)
    return module

def remove_cp_dropout(module):
    """Removes the tensor dropout from a TensorModule 

    Parameters
    ----------
    module : tltorch.TensorModule
        the tensor module parametrized by the tensor decomposition to which to apply tensor dropout
    
    Examples
    --------
    >>> trl = tltorch.CPTRL((10, 10), (10, ), rank='same')
    >>> trl = cp_dropout(trl, p=0.5)
    >>> remove_cp_dropout(trl)  
    """
    for key, hook in module._decomposition_forward_pre_hooks.items():
        if isinstance(hook, CPDropout):
            del module._decomposition_forward_pre_hooks[key]
            break

def tt_dropout(module, p):
    """TT Dropout

    Parameters
    ----------
    module : tltorch.TensorModule
        the tensor module parametrized by the tensor decomposition to which to apply tensor dropout
    p : float
        dropout probability
        if 0, no dropout is applied
        if 1, all the components but 1 are dropped in the latent space

    Returns
    -------
    TensorModule
        the module to which tensor dropout has been attached

    Examples
    --------
    >>> trl = tltorch.TensorTrainTRL((10, 10), (10, ), rank='same')
    >>> trl = tt_dropout(trl, p=0.5)
    >>> remove_tt_dropout(trl)
    """
    TTDropout.apply(module, p, min_dim=1)
    return module

def remove_tt_dropout(module):
    """Removes the tensor dropout from a TensorModule 

    Parameters
    ----------
    module : tltorch.TensorModule
        the tensor module parametrized by the tensor decomposition to which to apply tensor dropout
    
    Examples
    --------
    >>> trl = tltorch.TensorTrainTRL((10, 10), (10, ), rank='same')
    >>> trl = tt_dropout(trl, p=0.5)
    >>> remove_tt_dropout(trl)  
    """
    for key, hook in module._decomposition_forward_pre_hooks.items():
        if isinstance(hook, TTDropout):
            del module._decomposition_forward_pre_hooks[key]
            break

