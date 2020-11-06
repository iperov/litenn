import traceback
import numpy as np

import litenn as nn
import litenn.core as nc
from litenn.core import CLKernelHelper as ph

def transpose(input_t, axes_order):
    """
    Transpose operator

        axes_order     Int
                       Iterable of ints
                       None
    """
    return transpose_op(input_t, axes_order)

def depth_to_space(input_t, size):
    """
    Transpose channel space to spatial space

        size        int >= 2
                    channels must be divisible by size^2
    """

    N,C,H,W = input_t.shape
    if size < 2:
        raise ValueError('size should be >= 2')
    if C % (size*size) != 0:
        raise ValueError(f'Unable to divide {C} channels by {size*size} !')

    OC = C // (size*size)

    return ( input_t.reshape  ( (N, size, size, OC, H, W) )
                    .transpose( (0, 3, 4, 1, 5, 2) )
                    .reshape  ( (N, OC, H * size, W * size) ) )

def space_to_depth(input_t, size):
    """
    Transpose spatial space to channel space

     size   int >= 2
            spatial axes must be divisible by size
    """

    N,C,H,W = input_t.shape
    if size < 2:
        raise ValueError('size should be >= 2')
    if H % size != 0:
        raise ValueError(f'Unable to divide {H} H by {size} !')
    if W % size != 0:
        raise ValueError(f'Unable to divide {W} W by {size} !')

    OH, OW = H // size, W // size

    return (input_t.reshape  ( (N, C, OH, size, OW, size) )
                   .transpose( (0, 3, 5, 1, 2, 4) )
                   .reshape  ( (N, (size*size) * C, OH, OW) ))

def transpose_op(input_t, axes_order, output_t=None, is_add_to_output=False):
    """
    arguments:

        output_t            compute result to this Tensor.
                            Tensor may be with different shape, but should match total size.
                            gradfn will not be set.

        is_add_to_output    add result to output_t if output_t is set.
    """

    is_add_to_output = False if output_t is None else is_add_to_output

    op = nc.Cacheton.get(_TransposeOp, input_t.shape, nc.TensorAxes(axes_order), is_add_to_output)
    if output_t is None:
        output_t = nn.Tensor (op.info.output_shape)
        output_t._set_op_name('transpose')
        output_t._assign_gradfn (input_t, lambda O_t, dO_t: transpose_dI_gradfn(op, input_t, O_t, dO_t) )
    elif output_t.shape.size != op.info.output_shape.size:
        raise ValueError(f'output_t must have size {op.info.output_shape.size}')

    op.forward_krn.run(input_t, output_t)

    return output_t

def transpose_dI_gradfn(op, input, O, dO):
    # Transpose backward is forward but with "inversed" axes order
    nc.op.transpose( dO, op.axes_order.inversed(), output_t=input.get_grad(), is_add_to_output=True)

class _TransposeOp:
    def __init__(self, input_shape : nc.TensorShape, axes_order : nc.TensorAxes, is_add_to_output):
        self.axes_order = axes_order
        self.info = info = nc.info.InfoTranspose(input_shape, axes_order)
   
        self.forward_krn = nc.CLKernel(global_shape=(input_shape.size,), kernel_text=f"""
{ph.define_axes_accessor('I', input_shape)}
{ph.define_axes_accessor('O', info.output_shape)}
__kernel void impl(__global const float* I, __global float* O)
{{
    size_t gid = get_global_id(0);
    {ph.axes_idxs_from_var('i', input_shape.rank, 'gid')}
    O[O_idx({ph.axes_order_enum('i', axes_order )})]
    {'+=' if is_add_to_output else '='} I[gid];
}}""")


def transpose_test():
    for _ in range(10):
        for shape_len in range(2, 5):
            try:
                shape = np.random.randint( 8, size=(shape_len,) )+1
                axes_order = np.array([*range(shape_len)])
                np.random.shuffle(axes_order)

                val_n = np.random.randint( 2**8, size=shape ).astype(np.float32)
                transposed_n = np.transpose(val_n, axes_order)
                val_t = nn.Tensor_from_value(val_n)
                transposed_t = nn.transpose (val_t, axes_order )

                if transposed_n.shape != transposed_t.shape:
                    raise Exception('shape is not equal')
                if not all ( np.ndarray.flatten( transposed_t.np() == transposed_n ) ):
                    raise Exception(f'data is not equal {shape} {axes_order}')

                transposed_n_grad = np.random.randint( 2**8, size=transposed_n.shape ).astype(np.float32)

                val_t.get_grad().fill(1.0)  # Check addition to gradient
                nn.backward( {transposed_t:transposed_n_grad} , grad_for_non_trainables=True )

                if not all ( np.ndarray.flatten(np.transpose(val_t.get_grad().np()-1.0, axes_order) == transposed_n_grad) ):
                    raise Exception(f'grad is not equal')
            except:
                raise Exception(f"""
shape              : {shape}
axes_order         : {axes_order}
transposed_n_shape : {transposed_n.shape}
transposed_t_shape : {transposed_t.shape}
exception          : {traceback.format_exc()}
""")

        for size in [2,3,4]:
            try:
                N = 1+np.random.randint(8)
                C = (1+np.random.randint(8))*size*size
                H = W = (1+np.random.randint(8))*size

                shape = (N,C,H,W)

                val_n = np.random.randint( 2**8, size=shape ).astype(np.float32)
                val_t = nn.Tensor_from_value(val_n)

                d2s_val_t = depth_to_space(val_t, size)
                s2d_val_t = space_to_depth(d2s_val_t, size)

                if not all ( np.ndarray.flatten( val_t.np() == s2d_val_t.np() ) ):
                    raise Exception(f'data is not equal')
            except:
                raise Exception(f"""
depth_to_space/space_to_depth

shape              : {shape}
size               : {size}
exception          : {traceback.format_exc()}
""")
