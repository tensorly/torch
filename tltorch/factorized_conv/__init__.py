from ._tt_conv import TTConv, TTConvFactorized, TTConvReconstructed
from ._tucker_conv import TuckerConv, TuckerConvFactorized, TuckerConvFactorized
from ._cp_conv import (CPConv, CPConvFactorized,
                           CPConvReconstructed, CPConvMobileNet)
from ._base_conv import Conv1D, BaseFactorizedConv

# Register implementations of the Tucker Convolution
TuckerConv.register_implementation('factorized', TuckerConvFactorized, set_default=True)
TuckerConv.register_implementation('reconstructed', TuckerConvFactorized)

# Register implementations of the tt Convolution
TTConv.register_implementation('factorized', TTConvFactorized, set_default=True)
TTConv.register_implementation('reconstructed', TTConvReconstructed)

# Register implementations of the CP Convolution
CPConv.register_implementation('factorized', CPConvFactorized, set_default=True)
CPConv.register_implementation('reconstructed', CPConvReconstructed)
CPConv.register_implementation('mobilenet', CPConvMobileNet)