import math
import traceback
from collections import Iterable

import numpy as np

import litenn as nn
import litenn.core as nc
from litenn.core import CLKernelHelper as ph

# Save builtin slice() class
slice_cls = slice

def slice(input_t, slices):
    """
    slice operator.
    You should not use this operator directly. Use tensor[...values...]

        input_t     input tensor
        slices      argument received from class.__getitem__(slices)

    Remark.

    Slicing logic is not the same as numpy:
    For example np[2:0:1] slice will produce invalid array with zero index,
    but nn.slice() will select 2 index, same as val_t[2].
    """
    return slice_op(input_t, slices)

def slice_op(input_t, slices, output_t=None, is_add_to_output=False):
    """
    arguments:

        output_t            compute result to this Tensor.
                            Tensor may be with different shape, but should match total size.
                            gradfn will not be set.

        is_add_to_output    add result to output_t if output_t is set.
    """
    is_add_to_output = False if output_t is None else is_add_to_output

    op = nc.Cacheton.get(_SliceOp, input_t.shape, hashable_slices(slices), is_add_to_output )
    
    if output_t is None:
        if op.output_is_reshaped:
            return input_t.reshape(op.output_shape)
        else:        
            output_t = nn.Tensor (op.output_shape)
            output_t._set_op_name('slice')
            output_t._assign_gradfn (input_t, lambda O_t, dO_t: slice_dI_gradfn(op, input_t, dO_t) )
        
    elif output_t.shape.size != op.output_shape.size:
        raise ValueError(f'output_t must have size {op.output_shape.size}')
        
    op.forward_krn.run(input_t, output_t)

    return output_t
    


def slice_dI_gradfn(op, input, dO):
    op.backward_krn.run(input.get_grad(), dO)

class _SliceOp:
    def __init__(self, input_shape : nc.TensorShape, slices, is_add_to_output):

        # Validate slices argument for given shape.
        new_slices = []
        before_ellipsis = None

        for s in slices:
            if s is Ellipsis:
                before_ellipsis = new_slices
                new_slices = []
                continue
            elif s is not None and not isinstance(s, (int,tuple) ):
                raise ValueError(f'unknown slice argument {s} of type {s.__class__}')

            new_slices.append(s)

        if before_ellipsis is not None:
            # Process Ellipsis separator
            new_slices_n_axes = sum([ 1 for x in new_slices if x != None])
            before_ellipsis_n_axes = sum([ 1 for x in before_ellipsis if x != None])

            # Expand slices by filling intermediate (None,None,None) for each remaining axis
            new_slices = before_ellipsis + \
                         [(None,None,None)]*max(0, input_shape.rank-before_ellipsis_n_axes-new_slices_n_axes) + \
                         new_slices

        new_slices_n_axes = sum([ 1 for x in new_slices if x != None])
        if new_slices_n_axes > input_shape.rank:
            raise ValueError('slices arguments more than shape axes')
        elif new_slices_n_axes < input_shape.rank:
            # Fill remaining axes
            new_slices += [(None,None,None)]*( input_shape.rank - new_slices_n_axes )

        slices = tuple(new_slices)

        # Compute shapes
        output_is_reshaped = True  # Flag determines that output_tensor
                                   # can be just reshaped without any computation
        output_shape = []          # output tensor shape
        output_shape_krn = []      # output shape used in kernel, must match input shape
        input_axes_begin_step = [] # begin,step ints for every input shape

        i_axis = 0
        for v in slices:
            if v is None:
                # None is new axis
                # We can add unlimited number of (1,) axes at any place of shape
                output_shape.append(1)
                continue

            i_axis_size = input_shape[i_axis]
            i_axis += 1

            if isinstance(v, int):
                if v < 0: v += i_axis_size
                if v < 0 or v >= i_axis_size:
                    raise ValueError(f'index {v} is out of bounds for axis {i_axis} with size {i_axis_size}')
                b,e,s = v,v,1
            else:
                b,e,s = v

            # Fix begin, end, step values
            if s is None: s = 1
            if s == 0:
                raise ValueError('slice step cannot be zero')

            if b is None: b = 0 if s > 0 else i_axis_size-1
            if e is None: e = i_axis_size if s > 0 else -1
            elif e < 0: e += i_axis_size

            if b < 0: b += i_axis_size

            if s > 0:
                b = np.clip(b, 0, i_axis_size)
                e = np.clip(e, 0, i_axis_size)
            else:
                b = np.clip(b, 0, i_axis_size-1)
                e = np.clip(e, -1, i_axis_size)

            if i_axis_size != 1 and not (b == 0 and e == i_axis_size and s == 1):
                # Such params of axis slice will change input, thus output cannot be just reshaped input
                output_is_reshaped = False
                
            # Compute output_axis_size based on begin,end,step
            output_axis_size = max(0, math.ceil ( (e-b)/s ) )

            if output_axis_size >= 1:
                # >= 1 : select range of indexes, axis will remain
                output_shape.append(output_axis_size)
            # ^ othwerwise axis will be supressed

            # output_shape to use in kernel, must match rank of input shape
            output_shape_krn.append( max(1,output_axis_size) )

            # for every output_shape_krn axis
            # we have exact begin,step values to fetch value from input
            input_axes_begin_step.append ( (b,s) )

        output_shape_krn = nc.TensorShape(output_shape_krn)
        self.output_is_reshaped = output_is_reshaped
        self.output_shape = nc.TensorShape(output_shape) 


        self.forward_krn = nc.CLKernel(global_shape=(output_shape_krn.size,), kernel_text=f"""
{ph.define_axes_accessor('I', input_shape )}
{ph.define_axes_sizes('O', output_shape_krn )}
__kernel void impl(__global const float* I, __global float* O)
{{
size_t gid = get_global_id(0);
{ph.axes_idxs_from_var('o', output_shape_krn.rank, 'gid')}
{''.join( f'size_t i{i} = {b} + o{i} * {s};' for i, (b,s) in enumerate(input_axes_begin_step)  )  }
O[get_global_id(0)] {'+=' if is_add_to_output else '='} I[I_idx({ph.axes_seq_enum('i', input_shape.rank)})];
}}
""")

        self.backward_krn = nc.CLKernel(global_shape=(output_shape_krn.size,), kernel_text=f"""
{ph.define_axes_accessor('I', input_shape )}
{ph.define_axes_sizes('O', output_shape_krn )}
__kernel void impl(__global float* dI, __global const float* O)
{{
size_t gid = get_global_id(0);
{ph.axes_idxs_from_var('o', output_shape_krn.rank, 'gid')}
{''.join( f'size_t i{i} = {b} + o{i} * {s};' for i, (b,s) in enumerate(input_axes_begin_step)  )  }
dI[I_idx({ph.axes_seq_enum('i', input_shape.rank)})] += O[get_global_id(0)];
}}
""")

def hashable_slices(slices):
    # Convert slices to hashable arg.
    if not isinstance(slices, Iterable):
        slices = (slices,)
    normalized_slices = []
    for x in slices:
        if isinstance(x, slice_cls):
            normalized_slices.append( (x.start, x.stop, x.step) )
        elif x is Ellipsis or x is None:
            normalized_slices.append(x)
        else:
            normalized_slices.append(int(x))
    return tuple(normalized_slices)
    
def slice_test():
    for iteration in range(10):
        for shape_len in range(5,1,-1):
            try:
                while True:
                    shape = np.random.randint( 1, 8, size=(shape_len,) )

                    if iteration == 0:
                        slices = [ slice_cls(None,None,None), ] * shape_len
                        axis = np.random.randint(shape_len)
                        shape[axis] = 1
                        slices[axis] = 0                   
                    else:
                        slices = []
                        for i in range (shape_len):
                            axis_size = shape[i]
                            if np.random.randint(2) == 0:
                                v = axis_size-np.random.randint(axis_size*2)-1
                                slices.append (v)
                            else:
                                b = None if np.random.randint(2) == 0 else axis_size-np.random.randint(axis_size*2)
                                e = None if np.random.randint(2) == 0 else axis_size-np.random.randint(axis_size*2)
                                s = 1 if np.random.randint(2) == 0 else -1

                                slices.append ( slice_cls(b,e,s) )
                        
                        if np.random.randint(2) == 0:
                            axis = np.random.randint(shape_len)
                            slices[axis] = Ellipsis             
                                  
                    shape = tuple(shape)     
                    slices = tuple(slices)     
                    
                    
                    val_n = np.random.randint( 2**8, size=shape ).astype(np.float32)

                    sliced_n = val_n[slices]
                    val_t = nn.Tensor_from_value(val_n)
                    sliced_t = val_t[slices]

                    if 0 in sliced_n.shape:
                        # some cases like 0:1:-1 will produce zero shape and invalid array on numpy
                        # but nn.slice has no such behaviour, thus we have to generate new slice again
                        continue
                    
                    if np.prod(sliced_n.shape) != sliced_t.shape.size:
                        raise Exception(f'shape is not equal')

                    if not all ( np.ndarray.flatten( np.array(sliced_t.np()) ) == np.ndarray.flatten(np.array(sliced_n)) ):
                        raise Exception(f'data is not equal')

                    sliced_n_grad = np.random.randint( 2**8, size=sliced_n.shape ).astype(np.float32)
 
                    val_t.get_grad().fill(1.0)
                    nn.backward({ sliced_t : sliced_n_grad}, grad_for_non_trainables=True )
                    sliced_t_grad = np.array(val_t.get_grad().np()[slices])

                    if not all ( np.ndarray.flatten(np.array([sliced_t_grad-1.0])) == np.ndarray.flatten(np.array([sliced_n_grad])) ):
                        raise Exception(f'grad is not equal')

                    break
            except:
                raise Exception(f"""
shape          : {shape}
slices         : {slices}
sliced_n_shape : {sliced_n.shape}
sliced_t_shape : {sliced_t.shape}
exception      : {traceback.format_exc()}
""")
