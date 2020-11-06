"""
Welcome to litenn source code.

import litenn as nn  - is all user need.

Classes and Modules(Layers) are upper case
operators are lower case
Initializers in 'nn.initializer.' namespace
Optimizers in 'nn.optimizer.' namespace
"""
import litenn.core


from litenn.core.Tensor import (
                    Tensor,
                    Tensor_like,
                    Tensor_zeros_like,
                    Tensor_ones_like,
                    Tensor_from_value,
                    Tensor_sliced_from_value)

import litenn.core.info
import litenn.core.op

import litenn.core.initializer as initializer

from litenn.core.backward import backward

# Operators
from litenn.core.op.concat import concat
from litenn.core.op.conv2D import conv2D
from litenn.core.op.conv2DTranspose import conv2DTranspose
from litenn.core.op.depthwise_conv2D import depthwise_conv2D
from litenn.core.op.dropout import dropout
from litenn.core.op.dual_wise_op import (add,
                                        binary_crossentropy,
                                        categorical_crossentropy,
                                        sub,
                                        mul,
                                        div)
from litenn.core.op.element_wise_op import (abs,
                                       add_const,
                                       clip,
                                       cos,
                                       div_const,
                                       exp,
                                       mul_const,
                                       leaky_relu,
                                       log,
                                       relu,
                                       rdiv_const,
                                       rsub_const,
                                       sigmoid,
                                       sin,
                                       softmax,                                       
                                       sqrt,
                                       square,
                                       sub_const,
                                       tanh)
from litenn.core.op.matmul import matmul
from litenn.core.op.pool2D import (avg_pool2D,
                                   max_pool2D,
                                   min_pool2D)
from litenn.core.op.reduce import (moments,
                                   reduce_mean,
                                   reduce_min,
                                   reduce_max,
                                   reduce_std,
                                   reduce_sum,
                                   reduce_variance,)
from litenn.core.op.reshape import (flatten,
                                    reshape)
from litenn.core.op.resize2D_bilinear import resize2D_bilinear
from litenn.core.op.resize2D_nearest import resize2D_nearest
from litenn.core.op.slice import slice
from litenn.core.op.spatial_affine_transform2D import spatial_affine_transform2D
from litenn.core.op.spatial_transform2D import spatial_transform2D
from litenn.core.op.ssim import ssim, dssim
from litenn.core.op.stack import stack
from litenn.core.op.transpose import (depth_to_space,
                                      space_to_depth,
                                      transpose)

from litenn.core.op.tile import tile
from litenn.core.op.unfold2D import unfold2D

# Forward devices to litenn namespace
from litenn.core import devices

# forward Modules to litenn namespace
from litenn.core.module.Module import Module
from litenn.core.module.BatchNorm2D import BatchNorm2D
from litenn.core.module.Conv2D import Conv2D
from litenn.core.module.Conv2DTranspose import Conv2DTranspose
from litenn.core.module.Dense import Dense
from litenn.core.module.DenseAffine import DenseAffine
from litenn.core.module.Dropout import Dropout
from litenn.core.module.InstanceNorm2D import InstanceNorm2D
from litenn.core.module.SeparableConv2D import SeparableConv2D

import litenn.core.hint as hint

# Optimizers
import litenn.core.optimizer as optimizer

# Misc
import litenn.core.test as test

def cleanup():
    """
    Frees cached data, cached Tensors, memory of all current devices, resets hints.
    Tensor object must be zero.
    """
    litenn.core.Cacheton._cleanup()
    if Tensor._object_count != 0:
        raise Exception(f'Unable to cleanup while {Tensor._object_count} Tensor objects exist.')
    devices.cleanup()