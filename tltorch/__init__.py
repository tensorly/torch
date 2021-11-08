__version__ = '0.3.0'

from . import utils
from . import factorized_tensors
from .factorized_tensors import init
from . import functional
from . import factorized_layers

from .factorized_layers import FactorizedLinear, FactorizedConv, TRL, TCL, FactorizedEmbedding
from .factorized_tensors import FactorizedTensor, CPTensor, TTTensor, TuckerTensor, tensor_init
from .factorized_tensors import (TensorizedTensor, CPTensorized, BlockTT,
                                  TuckerTensorized)
from .tensor_hooks import (tensor_lasso, remove_tensor_lasso,
                           tensor_dropout, remove_tensor_dropout)
