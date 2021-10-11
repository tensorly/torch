from torch import nn
import torch

class FactorList(nn.Module):
    def __init__(self, parameters=None):
        super().__init__()
        self.keys = []
        self.counter = 0
        if parameters is not None:
            self.extend(parameters)
    
    def _unique_key(self):
        """Creates a new unique key"""
        key = f'factor_{self.counter}'
        self.counter += 1
        return key
        
    def append(self, element):
        key = self._unique_key()
        setattr(self, key, element)
        self.keys.append(key)
        
    def insert(self, index, element):
        key = self._unique_key()
        setattr(self ,key, element)
        self.keys.insert(index, key)
    
    def pop(self, index=-1):
        item = self[index]
        self.__delitem__(index)
        return item
    
    def __getitem__(self, index):
        keys = self.keys[index]
        if isinstance(keys, list):
            return self.__class__([getattr(self, key) for key in keys])
        return getattr(self, keys)
    
    def __setitem__(self, index, value):
        setattr(self, self.keys[index], value)
    
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
    
    def __add__(self, parameters):
        instance = self.__class__(self)
        instance.extend(parameters)
        return instance

    def __radd__(self, parameters):
        instance = self.__class__(parameters)
        instance.extend(self)
        return instance
    
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
            return self.__class__([getattr(self, key) for key in keys])
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
