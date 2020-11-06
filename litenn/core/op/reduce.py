import math
import traceback

import numpy as np

import litenn as nn
import litenn.core as nc
from litenn.core import CLKernelHelper as ph

def reduce_sum_op (input_t, axes=None, keepdims=False, output_t=None, is_add_to_output=False):
    """
    Reduce sum operator.

        input_t     Tensor

        axes(None)  int
                    Iterable of ints.
                    None - all axes

        keepdims(False)     keep reduced axes
    """
    return reduce_op ('sum', input_t, axes=axes, keepdims=keepdims, output_t=output_t, is_add_to_output=is_add_to_output)
def reduce_sum (input_t, axes=None, keepdims=False):
    """
    Reduce sum operator.

        input_t     Tensor

        axes(None)  int
                    Iterable of ints.
                    None - all axes

        keepdims(False)     keep reduced axes
    """
    return reduce_sum_op (input_t, axes=axes, keepdims=keepdims)

def reduce_mean_op (input_t, axes=None, keepdims=False, output_t=None, is_add_to_output=False):
    """
    Reduce mean operator.

        input_t     Tensor

        axes(None)  int
                    Iterable of ints.
                    None - all axes

        keepdims(False)     keep reduced axes
    """
    return reduce_op ('mean', input_t, axes=axes, keepdims=keepdims, output_t=output_t, is_add_to_output=is_add_to_output)
def reduce_mean(input_t, axes=None, keepdims=False):
    """
    Reduce mean operator.

        input_t     Tensor

        axes(None)  int
                    Iterable of ints.
                    None - all axes

        keepdims(False)     keep reduced axes
    """
    return reduce_mean_op (input_t, axes=axes, keepdims=keepdims)

def reduce_min_op (input_t, axes=None, keepdims=False, output_t=None, is_add_to_output=False):
    """
    Reduce min operator.

        input_t     Tensor

        axes(None)  int
                    Iterable of ints.
                    None - all axes

        keepdims(False)     keep reduced axes
    """
    return reduce_op ('min', input_t, axes=axes, keepdims=keepdims, output_t=output_t, is_add_to_output=is_add_to_output)
def reduce_min (input_t, axes=None, keepdims=False):
    """
    Reduce min operator.

        input_t     Tensor

        axes(None)  int
                    Iterable of ints.
                    None - all axes

        keepdims(False)     keep reduced axes
    """
    return reduce_min_op (input_t, axes=axes, keepdims=keepdims)

def reduce_max_op (input_t, axes=None, keepdims=False, output_t=None, is_add_to_output=False):
    """
    Reduce max operator.

        input_t     Tensor

        axes(None)  int
                    Iterable of ints.
                    None - all axes

        keepdims(False)     keep reduced axes
    """
    return reduce_op ('max', input_t, axes=axes, keepdims=keepdims, output_t=output_t, is_add_to_output=is_add_to_output)
def reduce_max (input_t, axes=None, keepdims=False):
    """
    Reduce max operator.

        input_t     Tensor

        axes(None)  int
                    Iterable of ints.
                    None - all axes

        keepdims(False)     keep reduced axes
    """
    return reduce_max_op (input_t, axes=axes, keepdims=keepdims)

def reduce_std(input_t, axes=None, keepdims=False):
    """
    Reduce std operator.

        input_t     Tensor

        axes(None)  int
                    Iterable of ints.
                    None - all axes

        keepdims(False)     keep reduced axes
    """
    return nn.sqrt(reduce_variance(input_t, axes, keepdims))


def reduce_variance(input_t, axes=None, keepdims=False):
    """
    Reduce variance operator.

        input_t     Tensor

        axes(None)  int
                    Iterable of ints.
                    None - all axes

        keepdims(False)     keep reduced axes
    """
    mean = reduce_mean(input_t, axes, keepdims=True)    
    return reduce_mean(nn.square(input_t - mean), axes, keepdims)
    
def moments(input_t, axes=None, keepdims=False):    
    """
    Returns (mean, variance) of input_t
    
        input_t     Tensor

        axes(None)  int
                    Iterable of ints.
                    None - all axes

        keepdims(False)     keep reduced axes
    """
    mean = reduce_mean(input_t, axes, keepdims)    
    mean_shape_keepdims = mean._op.info.output_shape_keepdims    
    var = reduce_mean(nn.square(input_t - mean.reshape(mean_shape_keepdims) ), axes, keepdims)
    return mean, var


def reduce_op (op_type, input_t, axes=None, keepdims=False, output_t=None, is_add_to_output=False):
    """
    arguments

        op_type     'sum' 'mean' 'min' 'max'

        output_t            compute result to this Tensor.
                            Tensor may be with different shape,
                            but should match total size.
                            gradfn will not be set.

        is_add_to_output    add result to output_t if output_t is set.
    """

    op = nc.Cacheton.get(_ReduceOp, op_type, input_t.shape, nc.TensorAxes(axes, input_t.shape.rank), keepdims)

    if output_t is None:
        output_t = nn.Tensor ( op.info.output_shape )
        output_t._op = op
        output_t._set_op_name (f'reduce_{op_type}')
        output_t._assign_gradfn ( input_t, lambda O_t, dO_t: reduce_dI_gradfn(op, input_t, O_t, dO_t, ) )
    elif output_t.shape.size != op.info.output_shape.size:
        raise ValueError(f'output_t must have size {op.info.output_shape.size}')

    # Make an intermediate tensor
    input_t_inter = nc.op.transpose(input_t, op.intermediate_transpose_axes)

    # Perform multistage inplace operation in intermediate tensor
    for stage, (shape, STAGE_COLS, STAGE_VALID_COLS) in enumerate(zip(op.forward_krn_shapes, op.forward_krn_stage_cols, op.forward_krn_stage_valid_cols)):
        op.forward_krn.run(input_t_inter, np.int64(op.COLS), np.int64(STAGE_COLS), np.int64(STAGE_VALID_COLS),
                           global_shape=shape)

    if op_type == 'mean':
        # divide values in ROWS by number of COLS
        _ReduceOp.mean_div_forward_krn.run(input_t_inter, np.int64(op.COLS), global_shape=(op.ROWS,) )

    # Fetch final tensor from zero indexes using slices argument
    nc.op.slice(input_t_inter, op.inter_slices, output_t=output_t, is_add_to_output=is_add_to_output)

    return output_t

def reduce_dI_gradfn(op, input_t, O_t, dO_t):
    if op.op_type in ['sum','mean']:
        op.backward_krn.run (input_t.get_grad(), dO_t, global_shape=op.krn_I_shape)
    elif op.op_type in ['max','min']:
        # Backward implementation is different from tensorflow in case when the same values exist in reduction axes.
        # Example tf     : 3 4 5 5 , max = 5, gradients : 0 0 0.5 0.5
        # Example litenn : 3 4 5 5 , max = 5, gradients : 0 0   1   0
        op.backward_krn.run(input_t.get_grad(), input_t, dO_t, O_t,
                            nn.Tensor_zeros_like(O_t), # used as memory for 'Lock'
                            global_shape=op.krn_I_shape)

class _ReduceOp:
    def __init__(self, op_type, input_shape : nc.TensorShape, axes : nc.TensorAxes, keepdims=False):
        self.op_type = op_type
        self.info = info = nc.info.InfoReduction(input_shape, axes, keepdims)

        # Determine transpose order for intermediate tensor, where reduction axes will be at the end
        self.intermediate_transpose_axes = info.output_axes + info.reduction_axes
        self.intermediate_shape = nc.info.InfoTranspose(input_shape, self.intermediate_transpose_axes).output_shape

        # slices argument to fetch processed tensor from zero indexes
        self.inter_slices = ( slice(None,None,None), ) * info.output_axes.rank + (0,) * info.reduction_axes.rank

        # COLS are reduction axes, ROWS are remaining axes
        rows_rank = info.output_axes.rank
        self.ROWS = ROWS = self.intermediate_shape[:rows_rank].size
        self.COLS = COLS = self.intermediate_shape[rows_rank:].size

        # Number of stages to operate COLS
        n_stages = (COLS-1).bit_length()
        self.forward_krn_shapes           = [ (ROWS * math.ceil(COLS/ (2**(stage+1)) ),) for stage in range(n_stages) ]
        self.forward_krn_stage_cols       = [ math.ceil(COLS / (2**(stage+1)) ) for stage in range(n_stages) ]
        self.forward_krn_stage_valid_cols = [ math.ceil(COLS / (2** stage   ) ) for stage in range(n_stages) ]

        self.krn_I_shape = (input_shape.size,)


        if op_type == 'mean':
            self.forward_krn = _ReduceOp.forward_krns['sum']
        else:
            self.forward_krn = _ReduceOp.forward_krns[op_type]

        if op_type in ['sum', 'mean']:
            self.backward_krn = nc.CLKernel(f"""
{ph.define_axes_accessor('I', input_shape)}
{ph.define_axes_accessor('O', info.output_shape_keepdims)}
__kernel void impl(__global float* dI, __global const float* dO)
{{
size_t gid = get_global_id(0);
{ph.axes_idxs_from_var('i', input_shape.rank, 'gid')}
dI[gid] += dO[O_idx_mod({ph.axes_seq_enum('i', input_shape.rank )})]
                        {f'/ {COLS}' if op_type == 'mean' else ''};
}}
""")
        elif op_type in ['min', 'max']:
            self.backward_krn = nc.CLKernel(f"""
{ph.define_axes_accessor('I', input_shape)}
{ph.define_axes_accessor('O', info.output_shape_keepdims)}
__kernel void impl(__global float* dI, __global const float* I, __global const float* dO, __global const float* O, __global const float* OLock)
{{
size_t gid = get_global_id(0);
{ph.axes_idxs_from_var('i', input_shape.rank, 'gid')}

size_t o_idx = O_idx({ph.axes_seq_enum('i', info.output_shape_keepdims.rank, zero_axes=info.reduction_axes)});

if (I[gid] == O[o_idx])
    dI[gid] += (atomic_inc( (volatile __global int*) &OLock[o_idx]) == 0) //1 if we are first
               * dO[o_idx];
}}
""")

    # Static kernels for any shapes
    forward_krns = { op_type : nc.CLKernel(f"""
#define fsum(a,b) a+b
__kernel void impl(__global float* I, long COLS, long STAGE_COLS, long STAGE_VALID_COLS)
{{
    size_t gid = get_global_id(0);

    size_t col = gid % STAGE_COLS;
    size_t row = gid / STAGE_COLS;
    size_t i_idx = row*COLS + col;

    size_t other_col = col + STAGE_COLS;
    if (other_col < STAGE_VALID_COLS)
        I[i_idx] = f{op_type} (I[i_idx], I[row*COLS + other_col]);
}}
""") for op_type in ['sum', 'min', 'max'] }

    mean_div_forward_krn = nc.CLKernel(f"""
__kernel void impl(__global float* I, long COLS)
{{
    size_t row = get_global_id(0);
    I[row*COLS] /= COLS;
}}
""")

def reduce_test():
    for _ in range(10):
        for op_type in ['sum', 'mean', 'min', 'max']:
            for shape_len in range(2, 5):
                try:
                    shape = np.random.randint( 8, size=(shape_len,) )+1

                    reduction_axes = np.array([*range(shape_len)])
                    np.random.shuffle(reduction_axes)

                    # Cut random amount of reduction_axes
                    reduction_axes = tuple(reduction_axes [:np.random.randint(shape_len+1)])
                    if len(reduction_axes) == 0:
                        reduction_axes = None

                    keepdims = np.random.randint(2) == 0

                    value_n = np.random.randint( 2**8, size=shape ).astype(np.float32)
                    value_t = nn.Tensor_from_value(value_n)

                    if op_type == 'sum':
                        reducted_t = value_t.sum(reduction_axes, keepdims=keepdims)
                        reducted_n = value_n.sum(reduction_axes, keepdims=keepdims)

                        reducted_n_keepdims_shape = value_n.sum(reduction_axes, keepdims=True).shape
                    elif op_type == 'mean':
                        reducted_t = value_t.mean(reduction_axes, keepdims=keepdims)
                        reducted_n = value_n.mean(reduction_axes, keepdims=keepdims)

                        reducted_n_keepdims_shape = value_n.mean(reduction_axes, keepdims=True).shape
                    elif op_type == 'max':
                        reducted_t = value_t.max(reduction_axes, keepdims=keepdims)
                        reducted_n = value_n.max(reduction_axes, keepdims=keepdims)

                        reducted_n_keepdims_shape = value_n.max(reduction_axes, keepdims=True).shape
                    elif op_type == 'min':
                        reducted_t = value_t.min(reduction_axes, keepdims=keepdims)
                        reducted_n = value_n.min(reduction_axes, keepdims=keepdims)

                        reducted_n_keepdims_shape = value_n.min(reduction_axes, keepdims=True).shape

                    if sum (np.ndarray.flatten( reducted_t.np() - reducted_n)) >= 1.0:
                        raise Exception(f'data is not equal')

                    value_t.get_grad().fill(1.0)

                    reducted_n_grad = np.random.randint( 2**8, size=reducted_n.shape ).astype(np.float32)
                    nn.backward( {reducted_t:reducted_n_grad}, grad_for_non_trainables=True )

                    if op_type == 'sum':
                        value_n_grad = np.broadcast_to( np.reshape(reducted_n_grad, reducted_n_keepdims_shape), value_n.shape )

                        if sum (np.ndarray.flatten( value_t.get_grad().np()-1.0 - value_n_grad )) >= 1.0:
                            raise Exception(f'dI is not equal')
                except:
                    raise Exception(f"""
op_type           : {op_type}
shape             : {shape}
reduction_axes    : {reduction_axes}
keepdims          : {keepdims}
reducted_n.shape  : {reducted_n.shape}
reducted_t.shape  : {reducted_t.shape}

exception : {traceback.format_exc() }
""")
