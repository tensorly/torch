import warnings

from torch import nn

# Author: Jean Kossaifi
# License: BSD 3 clause


class MetaFactorizedTensor(type):
    """Meta class for tensor factorizations
    
    .. info::
    
        1. Calls __new__ normally.
        2. Removes the keyword argument 'factorization' if present
        3. Calls __init__ with the remaining *args and **kwargs
    
    Why are we using this?
    ----------------------
    
    Tensor Factorization does not create its own instances.
    Instead, it defers to children class which do not take factorization as a parameter.
    
    We want to be able to create (e.g. CP) tensors in two ways:
    1. Indirectly: ``FactorizedTensor('cp', shape, rank)``
    2. Directly:   ``CP(shape, rank)``
    
    Note that in the second case, we don't want users to have to specify the 
    factorization, it would be redundant to ask them to create a CP as
    ``CP(shape, rank, factorization='CP')``.
    
    This means we need to intercept the call to __init__ and remove the factorization parameter
    when creating an instance from FactorizedTensor. Hence this metaclass.
        
    Current solution
    ----------------
    
    This metaclass customizes the object creation process.
    
    In the metaclass
    ++++++++++++++++
    
    First, we call __new__ with all the *args and **kwargs
    Then, if we are in FactorizedTensor, we remove the first argument.
    This is because FactorizedTensor never uses factorization in its own init.
    
    In __new__
    ++++++++++
    
    If `cls` is FactorizedTensor, we actually replace `cls` by one of the subclasses depending on
    the value of factorization and so create an instance of that subclass.
    If `cls` is already a subclass, we just create an instance of that.
    
    Creating a factorized tensor through `FactorizedTensor`
    ----------------------------------------------------------
    
    When creating a FactorizedTensor, the calls are as follow:
    1. __call__(FactorizedTensor, *args, **kwargs)
       where args = [factorization, *rest_of_args]
       
    2. __call__ first calls FactorizedTensor.__new__(FactorizedTensor, factorization, *args, **kwargs)
       
       In FactorizedTensor.__new__, instead of creating a new instance, we check for factorization's value
       against the internal _factorization dict that we maintain and return
       a new instance of FactorizedTensor._factorizations[factorization]
       
    3. We are now back in __call__ which now removes factorization from the argument list ``args``
       and calls instance.__init__ (now instance is CP, Tucker, **not** FactorizedTensor) with the
       remaining args and kwargs
    
    4. Since FactorizedTensor's signature is __init__(self, factorization, *args, **kwargs),
       the direct subclasses of FactorizedTensor call super().__init__(None, *args, **kwargs)
       
       This means that in practice FactorizedTensor always gets factorization=None.
       This does not matter as we only use factorization during the creation process.
       
       However, this forces users to specify factorization as a first argument when creating a tensor
       from Tensor Factorization.
       
    Creation through a subclass`FactorizedTensor`
    ------------------------------------------------
    Let's say now the user wants to directly create an instance of a subclass of `FactorizedTensor`,
    in this example, let's say `CP`.
    
    When creating a CPTensor, the calls are as follow:
    
    1. __call__(CPTensor, *args, **kwargs)
       __call__ just calls __new__, then __init__ with the given arguments and keyword arguments.
       
    2. __call__ first calls CPTensor.__new__(CPTensor, *args, **kwargs).
       In turn, this calls FactorizedTensor.__new__(CPTensor, *args, **kwargs)
       
       Since `cls` is now `CPTensor`, not `FactorizedTensor`, nothing special is done
       and ``super().__new__(cls, *args, **kwargs)`` is called to create an instance
       
    3. We are now back in __call__ again. Since `cls` is CPTensor and not FactorizedTensor,
       we just call instance.__init__
    
    4. Now, in CPTensor.__init__, we re-add the mendatory first arg `factorization` by calling super() as 
       ``super().__init__(self, None, *args, **kwargs)``
    """
    def __call__(cls, *args, **kwargs):
        instance = cls.__new__(cls, *args, **kwargs)
        kwargs.pop('factorization', None)

        instance.__init__(*args, **kwargs)
        return instance


class FactorizedTensor(nn.Module, metaclass=MetaFactorizedTensor):
    """Tensor Factorization

    .. important::

       All tensor factorization must have an `order` parameter
    """
    _factorizations = dict()
    
    def __init_subclass__(cls, name, **kwargs):
        """When a subclass is created, register it in _factorizations"""
        super().__init_subclass__(**kwargs)
        cls._factorizations[name.lower()] = cls
        cls._name = name

    def __new__(cls, *args, **kwargs):
        """Customize the creation of a factorized convolution

        Takes a parameter `factorization`, a string that specifies with subclass to use

        Returns
        -------
        FactorizedTensor._factorizations[factorization.lower()]
            subclass implementing the specified tensor factorization
        """
        if cls is FactorizedTensor:
            factorization = kwargs.get('factorization')
            try:
                cls = cls._factorizations[factorization.lower()]
            except KeyError:
                raise ValueError(f'Got factorization={factorization} but expected'
                                 f'one of {cls._factorizations.keys()}')
        
        instance = super().__new__(cls)

        return instance

    @classmethod
    def new(cls, shape, rank, factorization='CP', **kwargs):
        """Main way to create a factorized tensor


        Parameters
        ----------
        shape : tuple[int]
            shape of the factorized tensor to create
        rank : int, 'same' or float
            rank of the decomposition
        factorization : {'CP', 'TT', 'Tucker'}, optional
            Tensor factorization to use to decompose the tensor, by default 'CP'

        Returns
        -------
        TensorFactorization
            Tensor in Factorized form.

        Raises
        ------
        ValueError
            If the factorization given does not exist. 
        """
        try:
            cls = cls._factorizations[factorization.lower()]
        except KeyError:
            raise ValueError(f'Got factorization={factorization} but expected'
                             f'one of {cls._factorizations.keys()}')

        return cls.new(shape, rank, **kwargs)

    @classmethod
    def from_tensor(cls,tensor,  shape, rank, factorization='CP', **kwargs):
        """Create a factorized tensor by decomposing a full tensor


        Parameters
        ----------
        tensor : torch.tensor
            tensor to factorize
        shape : tuple[int]
            shape of the factorized tensor to create
        rank : int, 'same' or float
            rank of the decomposition
        factorization : {'CP', 'TT', 'Tucker'}, optional
            Tensor factorization to use to decompose the tensor, by default 'CP'

        Returns
        -------
        TensorFactorization
            Tensor in Factorized form.

        Raises
        ------
        ValueError
            If the factorization given does not exist. 
        """
        try:
            cls = cls._factorizations[factorization.lower()]
        except KeyError:
            raise ValueError(f'Got factorization={factorization} but expected'
                             f'one of {cls._factorizations.keys()}')

        return cls.from_tensor(tensor, shape, rank, **kwargs)

    def forward(self, indices=None, **kwargs):
        """To use a tensor factorization within a network, use ``tensor.forward``, or, equivalently, ``tensor()``

        Parameters
        ----------
        indices : int or tuple[int], optional
            use to index the tensor during the forward pass, by default None

        Returns
        -------
        TensorFactorization
            tensor[indices]
        """
        if indices is None:
            return self
        else:
            return self[indices]

    @property
    def _factorization(self, indices=None, **kwargs):
        """Returns the raw, unprocessed decomposition, same as `forward` but without forward hooks
        
        Parameters
        ----------
        indices : int, or tuple of int
            use to index the tensor
        
        Returns
        -------
        TensorFactorization
            tensor[indices]
        """
        if indices is None:
            return self
        else:
            return self[indices]

    def to_tensor(self):
        """Reconstruct the full tensor from its factorized form
        """ 
        raise NotImplementedError

    def dim(self):
        """Order of the tensor
        
        Notes
        -----
        fact_tensor.dim() == fact_tensor.ndim

        See Also 
        --------
        ndim
        """
        return len(self.shape)

    @property
    def ndim(self):
        """Order of the tensor
        
        Notes
        -----
        fact_tensor.dim() == fact_tensor.ndim

        See Also 
        --------
        dim
        """
        return len(self.shape)


    def size(self, index=None):
        """shape of the tensor

        Parameters
        ----------
        index : int, or tuple, default is None
            if not None, returns tensor.shape[index]

        See Also 
        --------
        shape
        """
        if index is None:
            return self.shape
        else:
            return self.shape[index]

    def normal_(self, mean=0, std=1):
        """Inialize the factors of the factorization such that the **reconstruction** follows a Gaussian distribution

        Parameters
        ----------
        mean : float, currently only 0 is supported
        std : float
            standard deviation
        
        Returns
        -------
        self
        """
        if mean != 0:
            raise ValueError(f'Currently only mean=0 is supported, but got mean={mean}')

    def __repr__(self):
        return f'{self.__class__.__name__}(shape={self.shape}, rank={self.rank})'

    def __torch_function__(self, func, types, args=(), kwargs=None):
        if kwargs is None:
            kwargs = {}

        args = [t.to_tensor() if hasattr(t, 'to_tensor') else t for t in args]
        return func(*args, **kwargs)



class TensorizedMatrix(nn.Module, metaclass=MetaFactorizedTensor):
    """Tensor representing one or a batch of tensorized vectors/matrices/tensors

    .. important::

       All tensor factorization must have an `order` parameter
    """
    _factorizations = dict()
    
    def __init_subclass__(cls, name, **kwargs):
        """When a subclass is created, register it in _factorizations"""
        super().__init_subclass__(**kwargs)
        cls._factorizations[name.lower()] = cls
        cls._name = name

    def __new__(cls, *args, **kwargs):
        """Customize the creation of a factorized convolution

        Takes as first parameter `factorization`, a string that specifies with subclass to use

        Returns
        -------
        FactorizedTensor._factorizations[factorization.lower()]
            subclass implementing the specified tensor factorization
        """
        if cls is TensorizedMatrix:
            factorization = kwargs.get('factorization')
            try:
                cls = cls._factorizations[factorization.lower()]
            except KeyError:
                raise ValueError(f'Got factorization={factorization} but expected'
                                 f'one of {cls._factorizations.keys()}')
        
        instance = super().__new__(cls)

        return instance

    @classmethod
    def new(cls, tensorized_row_shape, tensorized_column_shape, rank, n_matrices=(), factorization='CP', **kwargs):
        """Main way to create a Tensorized Matrix


        Parameters
        ----------
        tensorized_row_shape : tuple[int]
            The first dimension (rows) of the matrix will be tensorized to that shape
        tensorized_column_shape : tuple[int]
            The second dimension (columns) of the matrix will be tensorized to that shape
        rank : int, 'same' or float
            rank of the decomposition
        n_matrices : tuple or int, default is ()
            if not (), indicates how many matrices have to be jointly factorized
        factorization : {'CP', 'TT', 'Tucker'}, optional
            Tensor factorization to use to decompose the tensor, by default 'CP'

        Returns
        -------
        TensorizedMatrix
            Matrix in Tensorized and Factorized form.

        Raises
        ------
        ValueError
            If the factorization given does not exist. 
        """
        try:
            cls = cls._factorizations[factorization.lower()]
        except KeyError:
            raise ValueError(f'Got factorization={factorization} but expected'
                             f'one of {cls._factorizations.keys()}')
        return cls.new(tensorized_row_shape, tensorized_column_shape, rank, n_matrices=n_matrices, **kwargs)
    
    @classmethod
    def from_matrix(cls, matrix, tensorized_row_shape, tensorized_column_shape, rank, factorization='CP', **kwargs):
        """Create a Tensorized Matrix by tensorizing and decomposing an existing matrix


        Parameters
        ----------
        matrix : torch.tensor of order 2
            matrix to decompose
        tensorized_row_shape : tuple[int]
            The first dimension (rows) of the matrix will be tensorized to that shape
        tensorized_column_shape : tuple[int]
            The second dimension (columns) of the matrix will be tensorized to that shape
        rank : int, 'same' or float
            rank of the decomposition
        n_matrices : tuple or int, default is ()
            if not (), indicates how many matrices have to be jointly factorized
        factorization : {'CP', 'TT', 'Tucker'}, optional
            Tensor factorization to use to decompose the tensor, by default 'CP'

        Returns
        -------
        TensorizedMatrix
            Matrix in Tensorized and Factorized form.

        Raises
        ------
        ValueError
            If the factorization given does not exist. 
        """
        try:
            cls = cls._factorizations[factorization.lower()]
        except KeyError:
            raise ValueError(f'Got factorization={factorization} but expected'
                             f'one of {cls._factorizations.keys()}')
        return cls.from_matrix(matrix, tensorized_row_shape, tensorized_column_shape, rank, **kwargs)

    def forward(self, indices=None, **kwargs):
        """To use a tensor factorization within a network, use ``tensor.forward``, or, equivalently, ``tensor()``

        Parameters
        ----------
        indices : int or tuple[int], optional
            use to index the tensor during the forward pass, by default None

        Returns
        -------
        TensorizedMatrix
            tensor[indices]
        """
        if indices is None:
            return self
        else:
            return self[indices]

    @property
    def _factorization(self, indices=None, **kwargs):
        """Returns the raw, unprocessed decomposition, same as `forward` but without forward hooks
        
        Parameters
        ----------
        indices : int, or tuple of int
            use to index the tensor
        
        Returns
        -------
        TensorizedMatrix
            tensor[indices]
        """
        if indices is None:
            return self
        else:
            return self[indices]

    def to_tensor(self):
        """Reconstruct the full tensor from its factorized form
        """ 
        warnings.warn(f'{self} is being reconstructed into a full tensor, consider operating on the decomposed form.')
    
    def to_matrix(self):
        """Reconstruct the full matrix from the factorized tensorization

        If several matrices are parametrized, a batch of matrices is returned
        """ 
        warnings.warn(f'{self} is being reconstructed into a matrix, consider operating on the decomposed form.')

        full = self.to_tensor()
        if self.n_matrices == ():
            return full.reshape(self.shape)
        else:
            return full.reshape(self.n_matrices + self.shape)

    def dim(self):
        """Order of the tensor
        
        Notes
        -----
        fact_tensor.dim() == fact_tensor.ndim

        See Also 
        --------
        ndim
        """
        return len(self.shape)

    def size(self, index=None):
        """shape of the tensor

        Parameters
        ----------
        index : int, or tuple, default is None
            if not None, returns tensor.shape[index]

        See Also 
        --------
        shape
        """
        if index is None:
            return self.shape
        else:
            return self.shape[index]

    @property
    def ndim(self):
        """Order of the tensor
        
        Notes
        -----
        fact_tensor.dim() == fact_tensor.ndim

        See Also 
        --------
        dim
        """
        return len(self.shape)

    def normal_(self, mean, std):
        """Inialize the factors of the factorization such that the **reconstruction** follows a Gaussian distribution

        Parameters
        ----------
        mean : float, currently only 0 is supported
        std : float
            standard deviation
        
        Returns
        -------
        self
        """
        if mean != 0:
            raise ValueError(f'Currently only mean=0 is supported, but got mean={mean}')

    def __repr__(self):
        msg = f'{self.__class__.__name__}, shape={self.shape}, tensorized_row_shape={self.tensorized_row_shape}, '
        msg += f' tensorized_column_shape={self.tensorized_column_shape}, rank={self.rank}, '
        if self.n_matrices:
            msg += f'n_matrices={self.n_matrices} '
        msg += ')'
        return msg

    def __torch_function__(self, func, types, args=(), kwargs=None):
        if kwargs is None:
            kwargs = {}

        args = [t.to_matrix() if hasattr(t, 'to_tensor') else t for t in args]
        return func(*args, **kwargs)
    
