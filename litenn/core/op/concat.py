import traceback
import numpy as np
import litenn as nn
import litenn.core as nc
from litenn.core import CLKernelHelper as ph

def concat(tensor_list, axis):
    """
    Concat operator.

    arguments
    
        tensor_list     Iterable
    
        axis            Int

    Example:

        tensor_list [4,2],
                    [4,4] 
        axis    1
        result  [4,6]
    """
    return concat_op(tensor_list, axis)

def concat_op(tensor_list, axis, output_t=None, is_add_to_output=False):
    """
    arguments

        output_t            compute result to this Tensor.
                            Tensor may be with different shape,
                            but should match total size.
                            gradfn will not be set.

        is_add_to_output    add result to output_t if output_t is set.
    """
    is_add_to_output = False if output_t is None else is_add_to_output

    tensor_list = tuple(tensor_list)
    
    if not all (isinstance(tensor, nn.Tensor) for tensor in tensor_list):
        raise ValueError('All values must have type of Tensor')
    if len(tensor_list) == 0:
        raise ValueError('empty tensor_list')

    op = nc.Cacheton.get(_ConcatOp, tuple(t.shape for t in tensor_list), int(axis), is_add_to_output)
    
    if output_t is None:
        output_t = nn.Tensor (op.info.output_shape)
        output_t._set_op_name('concat')
        for n in range(len(tensor_list)):
            output_t._assign_gradfn (tensor_list[n], lambda O_t, dO_t, n=n: concat_gradfn(op, tensor_list[n], dO_t, n) )

    elif output_t.shape.size != op.info.output_shape.size:
        raise ValueError(f'output_t must have size {op.info.output_shape.size}')

    for i,t in enumerate(tensor_list):
        op.forward_krn.run (output_t, t, np.int64(op.info.axis_offsets[i]), np.int64(op.info.axis_sizes[i]), global_shape=(t.shape.size,) )

    return output_t

def concat_gradfn(op, input_t, dO_t, n):
    axis_offset = op.info.axis_offsets[n]
    axis_size = op.info.axis_sizes[n]
    axis_slice = slice(axis_offset, axis_offset+axis_size, 1)
    slices = (slice(None,None,None),)*op.info.axis + (axis_slice,) + (slice(None,None,None),)*(len(op.info.output_shape)-op.info.axis-1)
    nc.op.slice(dO_t, slices, output_t=input_t.get_grad(), is_add_to_output=True )


class _ConcatOp:
    def __init__(self, input_shapes, axis, is_add_to_output):
        self.info = info = nc.info.InfoConcat(input_shapes, axis)
        self.forward_krn = nc.CLKernel(f"""
{ph.define_axes_accessor('I', info.output_shape )}
{ph.define_axes_accessor('O', info.output_shape )}
#undef I{info.axis}
__kernel void impl(__global float* O, __global const float* I, long axis_offset, long I{info.axis})
{{
    size_t gid = get_global_id(0);
    {ph.axes_idxs_from_var('i', info.output_shape.rank, 'gid')}
    i{info.axis} += axis_offset;
    O[O_idx({ph.axes_seq_enum('i', info.output_shape.rank)})]
    {'+=' if is_add_to_output else '='} I[gid];
}}
""")

def concat_test():
    for _ in range(10):
        for shape_len in range(2, 5):
            try:
                shape = (np.random.randint( 8, size=(shape_len,) )+1).tolist()
                axis = np.random.randint(shape_len)
                count = np.random.randint(4)+1

                shapes = tuple( tuple( dim if i != axis else np.random.randint(8)+1
                                       for i,dim in enumerate(shape) )
                                for shape in ([shape] * count) )

                vals_n = [ np.random.randint( 2**8, size=shape ).astype(np.float32) for shape in shapes ]
                concat_n = np.concatenate(vals_n, axis)

                vals_t = [ nn.Tensor_from_value(vals_n[i]) for i in range(count) ]
                concat_t = nn.concat(vals_t, axis)

                concat_n_grad = np.random.randint( 2**8, size=concat_n.shape ).astype(np.float32)

                for t in vals_t:
                    t.get_grad().fill(1.0) # Check addition to gradient

                nn.backward( {concat_t:concat_n_grad} , grad_for_non_trainables=True )

                if not all ( np.ndarray.flatten( concat_t.np() == concat_n ) ):
                    raise Exception(f'data is not equal')

                axis_offset = 0
                for n in range(count):
                    axis_size = vals_n[n].shape[axis]
                    axis_slice = slice(axis_offset, axis_offset+axis_size, 1)
                    slices = (slice(None,None,None),)*axis + (axis_slice,) + (slice(None,None,None),)*(len(concat_n.shape)-axis-1)
                    axis_offset += axis_size

                    if not all ( np.ndarray.flatten(vals_t[n].get_grad().np()-1.0 == concat_n_grad[slices]) ):
                        raise Exception(f'grad is not equal')


            except:
                raise Exception(f"""
shape       : {shape}
axis        : {axis}
count       : {count}
exception   : {traceback.format_exc()}
""")