"""
litenn hints
"""

from litenn.core.module.Conv2D import hint_Conv2D_default_kernel_initializer as \
                                      Conv2D_default_kernel_initializer
                                      
from litenn.core.module.Conv2DTranspose import hint_Conv2DTranspose_default_kernel_initializer as \
                                               Conv2DTranspose_default_kernel_initializer

from litenn.core.module.Dense import hint_Dense_default_weight_initializer as \
                                     Dense_default_weight_initializer

from litenn.core.module.DepthwiseConv2D import hint_DepthwiseConv2D_default_depthwise_initializer as \
                                               DepthwiseConv2D_default_depthwise_initializer

from litenn.core.module.SeparableConv2D import (hint_SeparableConv2D_default_depthwise_initializer as \
                                                SeparableConv2D_default_depthwise_initializer,                                                 
                                                hint_SeparableConv2D_default_pointwise_initializer as \
                                                SeparableConv2D_default_pointwise_initializer )



