import tensorly as tl
tl.set_backend('pytorch')
from tensorly import random
from tensorly.testing import assert_

import numpy as np
import warnings
import torch
from torch import nn
from torch.nn import functional as F

import tltorch as tltorch

# Author: Jean Kossaifi
# License: BSD 3 clause


class TuckerL1Regularizer():
    """Decomposition Hook for Tensor Lasso on Tucker tensors

        Applies a generalized Lasso (l1 regularization) on the tensor layers the regularization it is applied to.


    Parameters
    ----------
    penalty : float, default is 0.01
        scaling factor for the loss
    
    clamp_weights : bool, default is True
        if True, the lasso weights are clamp between -1 and 1
    
    threshold : float, default is 1e-6
        if a lasso weight is lower than the set threshold, it is set to 0

    normalize_loss : bool, default is True  
        If True, the loss will be between 0 and 1.
        Otherwise, the raw sum of absolute weights will be returned.

    Examples
    --------

    First you need to create an instance of the regularizer:

    >>> regularizer = TuckerL1Regularizer(penalty=penalty)

    You can apply the regularizer to one or several layers:

    >>> trl = TRL((5, 5), (5, 5), rank='same')
    >>> trl2 = TRL((5, 5), (2, ), rank='same')
    >>> regularizer.apply(trl)
    >>> regularizer.apply(trl2)

    The lasso is automatically applied:

    >>> x = trl(x)
    >>> pred = trl2(x)
    >>> loss = your_loss_function(pred)

    Add the Lasso loss: 

    >>> loss = loss + regularizer.loss

    You can now backpropagate through your loss as usual:

    >>> loss.backwards()

    After you finish updating the weights, don't forget to reset the regularizer, 
    otherwise it will keep accumulating values!

    >>> loss.reset()

    You can also remove the regularizer with `regularizer.remove(trl)`.
    """
    _log = []
    
    def __init__(self, penalty=0.01, clamp_weights=True, threshold=1e-6, normalize_loss=True):
        self.penalty = penalty
        self.clamp_weights = clamp_weights
        self.threshold = threshold
        self.normalize_loss = normalize_loss

        # Initialize the counters
        self.reset()
        
    def reset(self):
        """Reset the loss, should be called at the end of each iteration.
        """
        self._loss = 0
        self.n_element = 0

    @property
    def loss(self):
        """Returns the current Lasso (l1) loss for the layers that have been called so far.

        Returns
        -------
        float
            l1 regularization on the tensor layers the regularization has been applied to.
        """
        if self.n_element == 0:
            warnings.warn('The L1Regularization was not applied to any weights.')
            return 0
        elif self.normalize_loss:
            return self.penalty*self._loss/self.n_element
        else:
            return self.penalty*self._loss

    def __call__(self, module, tucker_tensor):
        lasso_weights = getattr(module, 'lasso_weights')
        order = len(lasso_weights)

        with torch.no_grad():
            for i in range(order):
                if self.clamp_weights:
                    lasso_weights[i].data = torch.clamp(lasso_weights[i].data, -1, 1)

                if self.threshold:
                    lasso_weights[i] = F.threshold(lasso_weights[i], threshold=self.threshold, value=0, inplace=True)

            setattr(module, 'lasso_weights', lasso_weights)

        for weight in lasso_weights:
            self.n_element += weight.numel()
            self._loss = self._loss + torch.sum(torch.abs(weight))

        return self.apply_lasso(tucker_tensor, lasso_weights)

    def apply_lasso(self, tucker_tensor, lasso_weights):
        """Applies the lasso to a decomposed tensor
        """
        core, factors = tucker_tensor
        factors = [factor*w  for (factor, w) in zip(factors, lasso_weights)]
        return core, factors

    def apply(self, module):
        """Apply an instance of the L1Regularizer to a tensor module

        Parameters
        ----------
        module : TensorModule
            module on which to add the regularization

        Returns
        -------
        TensorModule (with Regularization hook)
        """
        if not isinstance(module, tltorch.TensorModule):
            raise ValueError(f'L1Regularizer can only be applied onto a TensorModule but got {module.__class__.__name__}.')

        rank = module.rank
        context = tl.context(module.core)
        lasso_weights = nn.ParameterList([nn.Parameter(torch.ones(r, **context)) for r in rank])
        setattr(module, 'lasso_weights', lasso_weights)
        handle = module.register_decomposition_forward_pre_hook(self, 'L1Regularizer')
        return module

    def remove(self, module):
        """Remove the Regularization from a module.
        """
        for key in module._decomposition_forward_pre_hooks:
            if key == 'L1Regularizer':
                delattr(module, 'lasso_weights')
                del module._decomposition_forward_pre_hooks[key]
            break
            

class TTL1Regularizer():
    """Decomposition Hook for Tensor Lasso on TT tensors

    Parameters
    ----------
    penalty : float, default is 0.01
        scaling factor for the loss
    
    clamp_weights : bool, default is True
        if True, the lasso weights are clamp between -1 and 1
    
    threshold : float, default is 1e-6
        if a lasso weight is lower than the set threshold, it is set to 0

    normalize_loss : bool, default is True  
        If True, the loss will be between 0 and 1.
        Otherwise, the raw sum of absolute weights will be returned.

    Examples
    --------

    First you need to create an instance of the regularizer:

    >>> regularizer = TTL1Regularizer(penalty=penalty)

    You can apply the regularizer to one or several layers:

    >>> trl = TensorTrainTRL((5, 5), (5, 5), rank='same')
    >>> trl2 = TensorTrainTRL((5, 5), (2, ), rank='same')
    >>> regularizer.apply(trl)
    >>> regularizer.apply(trl2)

    The lasso is automatically applied:

    >>> x = trl(x)
    >>> pred = trl2(x)
    >>> loss = your_loss_function(pred)

    Add the Lasso loss: 

    >>> loss = loss + regularizer.loss

    You can now backpropagate through your loss as usual:

    >>> loss.backwards()

    After you finish updating the weights, don't forget to reset the regularizer, 
    otherwise it will keep accumulating values!

    >>> loss.reset()

    You can also remove the regularizer with `regularizer.remove(trl)`.
    """
    def __init__(self, penalty=0.01, clamp_weights=True, threshold=1e-6, normalize_loss=True):
        self.penalty = penalty
        self.clamp_weights = clamp_weights
        self.threshold = threshold
        self.normalize_loss = normalize_loss

        # Initialize the counters
        self.reset()
        
    def reset(self):
        """Reset the loss, should be called at the end of each iteration.
        """
        self._loss = 0
        self.n_element = 0

    @property
    def loss(self):
        """Returns the current Lasso (l1) loss for the layers that have been called so far.

        Returns
        -------
        float
            l1 regularization on the tensor layers the regularization has been applied to.
        """
        if self.n_element == 0:
            warnings.warn('The L1Regularization was not applied to any weights.')
            return 0
        elif self.normalize_loss:
            return self.penalty*self._loss/self.n_element
        else:
            return self.penalty*self._loss

    def __call__(self, module, tt_tensor):
        lasso_weights = getattr(module, 'lasso_weights')
        order = len(lasso_weights)

        with torch.no_grad():
            for i in range(order):
                if self.clamp_weights:
                    lasso_weights[i].data = torch.clamp(lasso_weights[i].data, -1, 1)

                if self.threshold:
                    lasso_weights[i] = F.threshold(lasso_weights[i], threshold=self.threshold, value=0, inplace=True)

            setattr(module, 'lasso_weights', lasso_weights)

        for weight in lasso_weights:
            self.n_element += weight.numel()
            self._loss = self._loss + torch.sum(torch.abs(weight))

        return self.apply_lasso(tt_tensor, lasso_weights)

    def apply_lasso(self, tt_tensor, lasso_weights):
        """Applies the lasso to a decomposed tensor
        """
        factors = [factor*w  for (factor, w) in zip(tt_tensor, lasso_weights)] + [tt_tensor[-1]]
        return factors

    def apply(self, module):
        """Apply an instance of the L1Regularizer to a tensor module

        Parameters
        ----------
        module : TensorModule
            module on which to add the regularization

        Returns
        -------
        TensorModule (with Regularization hook)
        """
        rank = module.rank[1:-1]
        lasso_weights = nn.ParameterList([nn.Parameter(torch.ones(1, 1, r)) for r in rank])
        setattr(module, 'lasso_weights', lasso_weights)
        handle = module.register_decomposition_forward_pre_hook(self, 'L1Regularizer')
        return module

    def remove(self, module):
        """Remove the Regularization from a module.
        """
        for key, hook in module._decomposition_forward_pre_hooks.items():
            if key == 'L1Regularizer':
                delattr(module, 'lasso_weights')
                del module._decomposition_forward_pre_hooks[key]
            break
    

class CPL1Regularizer():
    """Decomposition Hook for Tensor Lasso on TT tensors

    Parameters
    ----------
    penalty : float, default is 0.01
        scaling factor for the loss
    
    clamp_weights : bool, default is True
        if True, the lasso weights are clamp between -1 and 1
    
    threshold : float, default is 1e-6
        if a lasso weight is lower than the set threshold, it is set to 0

    normalize_loss : bool, default is True  
        If True, the loss will be between 0 and 1.
        Otherwise, the raw sum of absolute weights will be returned.

    Examples
    --------

    First you need to create an instance of the regularizer:

    >>> regularizer = CPL1Regularizer(penalty=penalty)

    You can apply the regularizer to one or several layers:

    >>> trl = CPTRL((5, 5), (5, 5), rank='same')
    >>> trl2 = CPTRL((5, 5), (2, ), rank='same')
    >>> regularizer.apply(trl)
    >>> regularizer.apply(trl2)

    The lasso is automatically applied:

    >>> x = trl(x)
    >>> pred = trl2(x)
    >>> loss = your_loss_function(pred)

    Add the Lasso loss: 

    >>> loss = loss + regularizer.loss

    You can now backpropagate through your loss as usual:

    >>> loss.backwards()

    After you finish updating the weights, don't forget to reset the regularizer, 
    otherwise it will keep accumulating values!

    >>> loss.reset()

    You can also remove the regularizer with `regularizer.remove(trl)`.

    """
    def __init__(self, penalty=0.01, clamp_weights=True, threshold=1e-6, normalize_loss=True):
        self.penalty = penalty
        self.clamp_weights = clamp_weights
        self.threshold = threshold
        self.normalize_loss = normalize_loss

        # Initialize the counters
        self.reset()
        
    def reset(self):
        """Reset the loss, should be called at the end of each iteration.
        """
        self._loss = 0
        self.n_element = 0

    @property
    def loss(self):
        """Returns the current Lasso (l1) loss for the layers that have been called so far.

        Returns
        -------
        float
            l1 regularization on the tensor layers the regularization has been applied to.
        """
        if self.n_element == 0:
            warnings.warn('The L1Regularization was not applied to any weights.')
            return 0
        elif self.normalize_loss:
            return self.penalty*self._loss/self.n_element
        else:
            return self.penalty*self._loss

    def __call__(self, module, cp_tensor):
        """CP already includes weights, we'll just take their l1 norm
        """
        weights = getattr(module, 'weights')

        with torch.no_grad():
            if self.clamp_weights:
                weights.data = torch.clamp(weights.data, -1, 1)
                setattr(module, 'weights', weights)

            if self.threshold:
                weights.data = F.threshold(weights.data, threshold=self.threshold, value=0, inplace=True)
                setattr(module, 'weights', weights)

        self.n_element += weights.numel()
        self._loss = self._loss + self.penalty*torch.norm(weights, 1)
        return cp_tensor
            
    def apply(self, module):
        """Apply an instance of the L1Regularizer to a tensor module

        Parameters
        ----------
        module : TensorModule
            module on which to add the regularization

        Returns
        -------
        TensorModule (with Regularization hook)
        """
        handle = module.register_decomposition_forward_pre_hook(self, 'L1Regularizer')
        return module

    def remove(self, module):
        """Remove the Regularization from a module.
        """
        for key, hook in module._decomposition_forward_pre_hooks.items():
            if key == 'L1Regularizer':
                del module._decomposition_forward_pre_hooks[key]
            break
            


