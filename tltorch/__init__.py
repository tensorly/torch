__version__ = '0.1.0'

from ._tcl import TCL
from ._trl import TuckerTRL, TensorTrainTRL, CPTRL
from ._factorized_linear import CPLinear, TuckerLinear, TTLinear, TTMLinear
from .factorized_conv import Conv1D, BaseFactorizedConv
from .factorized_conv import CPConv, TuckerConv, TTConv
from ._tensor_dropout import TuckerDropout, CPDropout, TTDropout
from ._tensor_dropout import (cp_dropout, remove_cp_dropout,
                              tucker_dropout, remove_tucker_dropout,
                              tt_dropout, remove_tt_dropout)
from ._tensor_lasso import TuckerL1Regularizer, CPL1Regularizer, TTL1Regularizer
from .base import TensorModule