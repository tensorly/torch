import torch
from collections import OrderedDict
from torch import nn
import weakref


class RemovableHandle(object):
    """A handle which provides the capability to remove a hook.
    
    Adapted from Pytorch: torch.utils.hooks.py:7
    """
    def __init__(self, hooks_dict, name):
        # weakref will not keep the dict alive if there are no other references to it.
        self.hooks_dict_ref = weakref.ref(hooks_dict)
        self.name = name

    def remove(self):
        hooks_dict = self.hooks_dict_ref()
        if hooks_dict is not None and self.name in hooks_dict:
            del hooks_dict[self.name]


class TensorModule(nn.Module):
    """A PyTorch module augmented for tensor parametrization
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._decomposition_forward_pre_hooks = OrderedDict()

    def register_decomposition_forward_pre_hook(self, hook, name=None):
        """Attach a new hook to be applied to the decomposition parametrizing the layer, before the forward.

        Decomposition hooks are functions called before the forward pass
        that take as input the module and the decomposition
        and return a modified decomposition.

        Decomposition hooks must be function with the following signature::

            hook(module, decomposition) -> modified decomposition
        """
        if name is None:
            if hasattr(hook, 'name'):
                name = hook.name
            else:
                name = hook.__class__.__name__

        handle = RemovableHandle(self._decomposition_forward_pre_hooks, name)
        self._decomposition_forward_pre_hooks[name] = hook
        return handle

    def get_decomposition(self):
        """Returns the tensor decomposition parametrizing the layer
        """
        raise NotImplementedError()

    def _process_decomposition(self):
        """Applies all the decomposition_forward_pre_hooks before returning the decomposition

        This function should be used by all the Tensor layers to get their decomposition in the forward pass.
        """
        decomposition = self.get_decomposition()

        for hook in self._decomposition_forward_pre_hooks.values():
            decomposition = hook(self, decomposition)
        
        return decomposition


class ParameterList(nn.Module):
    def __init__(self, parameters=None):
        super().__init__()
        self.keys = []
        self.counter = 0
        if parameters is not None:
            self.extend(parameters)
    
    def _unique_key(self):
        """Creates a new unique key"""
        key = f'param_{self.counter}'
        self.counter += 1
        return key
        
    def append(self, element):
        # p = nn.Parameter(element)
        key = self._unique_key()
        self.register_parameter(key, element)
        self.keys.append(key)
        
    def insert(self, index, element):
        # p = nn.Parameter(element)
        key = self._unique_key()
        self.register_parameter(key, element)
        self.keys.insert(index, key)
    
    def pop(self, index=-1):
        item = self[index]
        self.__delitem__(index)
        return item
        
    def __getitem__(self, index):
        keys = self.keys[index]
        if isinstance(keys, list):
            #return self.__class__([getattr(self, key) for key in keys])
            params = [getattr(self, key) for key in keys]
            return self.__class__(params)
        return getattr(self, keys)
    
    def __setitem__(self, index, value):
        self.register_parameter(self.keys[index], value)
    
    def __delitem__(self, index):
        delattr(self, self.keys[index])
        self.keys.__delitem__(index)
        
    def __len__(self):
        return len(self.keys)
    
    def extend(self, parameters):
        for param in parameters:
            self.append(param)
    
    def __iadd__(self, parameters):
        return self.extend(parameters)
    
    def extra_repr(self) -> str:
        child_lines = []
        for k, p in self._parameters.items():
            size_str = 'x'.join(str(size) for size in p.size())
            device_str = '' if not p.is_cuda else ' (GPU {})'.format(p.get_device())
            parastr = 'Parameter containing: [{} of size {}{}]'.format(
                torch.typename(p), size_str, device_str)
            child_lines.append('  (' + str(k) + '): ' + parastr)
        tmpstr = '\n'.join(child_lines)
        return tmpstr
