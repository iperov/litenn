import traceback
import numpy as np
import litenn as nn
import litenn.core as nc
from litenn.core import CLKernelHelper as ph

def stack(tensor_list, axis):
    """
    Stack operator.
    
    arguments
    
        tensor_list     Iterable of Tensors
        
        axis            Int
        

    Stack list of tensors along new axis.

    Example:

        [4,4],[4,4], axis=0 produce [2,4,4]
        [4,4],[4,4], axis=1 produce [4,2,4]
        [4,4],[4,4], axis=2 produce [4,4,2]
    """
    return stack_op(tensor_list, axis)

def stack_op(tensor_list, axis, output_t=None, is_add_to_output=False):
    """
    arguments:

        output_t            compute result to this Tensor.
                            Tensor may be with different shape, but should match total size.
                            gradfn will not be set.

        is_add_to_output    add result to output_t if output_t is set.
    """
    is_add_to_output = False if output_t is None else is_add_to_output

    tensor_list = tuple(tensor_list)

    if not all (isinstance(tensor, nn.Tensor) for tensor in tensor_list):
        raise ValueError('All value must have type of Tensor')

    stack_count = len(tensor_list)
    if stack_count == 0:
        raise ValueError('tensor_list is empty')

    input_shape = tensor_list[0].shape

    if not all (tensor.shape == input_shape for tensor in tensor_list):
        raise ValueError('All tensors must have the same shape')

    op = nc.Cacheton.get(_StackOp, input_shape, int(axis), stack_count, is_add_to_output)
    
    if output_t is None:
        output_t = nn.Tensor (op.info.output_shape)
        output_t._set_op_name('stack')
        for n in range(stack_count):
            output_t._assign_gradfn (tensor_list[n], lambda O_t, dO_t, n=n: stack_gradfn(op, tensor_list[n], dO_t, n ))

    elif output_t.shape.size != op.info.output_shape.size:
        raise ValueError(f'output_t must have size {op.info.output_shape.size}')

    for i in range(stack_count):
        op.forward_krn.run(output_t, tensor_list[i], np.int64(i) )

    return output_t

def stack_gradfn(op, input_t, dO_t, n):
    slices = (slice(None,None,None),)*op.info.axis + (n,) + (slice(None,None,None),)*(len(op.info.output_shape)-op.info.axis-1)
    nc.op.slice(dO_t, slices, output_t=input_t.get_grad(), is_add_to_output=True )

class _StackOp:
    def __init__(self, input_shape : nc.TensorShape, axis, stack_count, is_add_to_output):
        self.info = info = nc.info.InfoStack(input_shape, axis, stack_count)

        self.forward_krn = nc.CLKernel(global_shape=(input_shape.size,), kernel_text=f"""
{ph.define_axes_accessor('I', input_shape )}
{ph.define_axes_accessor('O', info.output_shape )}
__kernel void impl(__global float* O, __global const float* I, long n)
{{
    size_t gid = get_global_id(0);
    {ph.axes_idxs_from_var('i', input_shape.rank, 'gid')}
    O[O_idx({ph.axes_seq_enum('i', input_shape.rank, new_axis=('n', info.axis))})]
    {'+=' if is_add_to_output else '='} I[gid];
}}
""")

def stack_test():
    for _ in range(10):
        for shape_len in range(1, 4):
            try:
                shape = tuple(np.random.randint( 8, size=(shape_len,) )+1)
                axis = np.random.randint(shape_len+1)
                stack_count = np.random.randint(4)+1

                vals_n = [ np.random.randint( 2**8, size=shape ).astype(np.float32) for i in range(stack_count) ]
                vals_t = [ nn.Tensor_from_value(vals_n[i]) for i in range(stack_count) ]
                stack_n = np.stack(vals_n, axis)
                stack_t = nn.stack(vals_t, axis)

                if not all ( np.ndarray.flatten( stack_t.np() == stack_n ) ):
                    raise Exception(f'data is not equal')

                stack_n_grad = np.random.randint( 2**8, size=stack_n.shape ).astype(np.float32)
                for val_t in vals_t:
                    val_t.get_grad().fill(1.0)
                nn.backward( {stack_t:stack_n_grad} , grad_for_non_trainables=True )

                for n in range(stack_count):
                    slices = (slice(None,None,None),)*axis + (n,) + (slice(None,None,None),)*(len(stack_n.shape)-axis-1)
                    if not all ( np.ndarray.flatten(vals_t[n].get_grad().np()-1.0 == stack_n_grad[slices]) ):
                        raise Exception(f'grad is not equal')

            except:
                raise Exception(f"""
shape         : {shape}
axis          : {axis}
stack_count   : {stack_count}
stack_n_shape : {stack_n.shape}
stack_t_shape : {stack_t.shape}
exception     : {traceback.format_exc()}
""")