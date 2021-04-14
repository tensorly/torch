__version__ = '0.2.0'

from . import utils
from . import tensor_factorizations
from .tensor_factorizations import init
from . import functional
from . import factorized_layers

from .factorized_layers import FactorizedLinear, FactorizedConv, TRL, TCL
from .tensor_factorizations import FactorizedTensor, CPTensor, TTTensor, TuckerTensor, tensor_init
from .tensor_factorizations import TensorizedMatrix, CPMatrix, TuckerMatrix, TTMatrix
from .tensor_hooks import (tensor_lasso, remove_tensor_lasso,
                           tensor_dropout, remove_tensor_dropout)
