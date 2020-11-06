import traceback
import numpy as np
import litenn as nn
import litenn.core as nc
from litenn.core import CLKernelHelper as ph

def tile(input_t, tiles):
    """
    Tile operator
    
    arguments
    
        tiles       Iterable of ints

    Example
    
        Tensor  (2,4,1)
        tiles   (2,2,1) 
        result  (4,8,1)
    """
    return tile_op(input_t, tiles)

def tile_op(input_t, tiles, output_t=None, is_add_to_output=False):
    """
    arguments:

        output_t            compute result to this Tensor.
                            Tensor may be with different shape, but should match total size.
                            gradfn will not be set.

        is_add_to_output    add result to output_t if output_t is set.
    """
    is_add_to_output = False if output_t is None else is_add_to_output
    op = nc.Cacheton.get(_TileOp, input_t.shape, tuple(int(tile) for tile in tiles), is_add_to_output)

    if output_t is None:
        output_t = nn.Tensor (op.info.output_shape)
        output_t._set_op_name('tile')
        output_t._assign_gradfn (input_t, lambda O_t, dO_t: tile_gradfn(op, input_t, dO_t) )
    elif output_t.shape.size != op.info.output_shape.size:
        raise ValueError(f'output_t must have size {op.info.output_shape.size}')

    op.forward_krn.run(output_t, input_t)

    return output_t

def tile_gradfn(op, input_t, dO_t):
    dI = input_t.get_grad()
    for axes_slice in op.info.axes_slices:
        op.backward_krn.run(dI, dO_t, *[ np.int64(sl.start) for sl in axes_slice])

class _TileOp:
    def __init__(self, input_shape : nc.TensorShape, tiles, is_add_to_output):
        self.info = info = nc.info.InfoTile(input_shape, tiles)

        self.forward_krn = nc.CLKernel(global_shape=(info.output_shape.size,), kernel_text=f"""
{ph.define_axes_accessor('I', input_shape)}
{ph.define_axes_accessor('O', info.output_shape)}
__kernel void impl(__global float* O, __global const float* I)
{{
    size_t gid = get_global_id(0);
    {ph.axes_idxs_from_var('o', info.output_shape.rank, 'gid')}
    O[gid] {'+=' if is_add_to_output else '='}
           I[I_idx_mod({ph.axes_seq_enum('o', info.output_shape.rank)})];
}}
""")
        self.backward_krn = nc.CLKernel(global_shape=(input_shape.size,), kernel_text=f"""
{ph.define_axes_accessor('I', input_shape)}
{ph.define_axes_accessor('O', info.output_shape)}
__kernel void impl(__global float* dI, __global const float* dO
                    ,{','.join([ f' long i{i}_offset' for i in range(input_shape.rank) ]) } )
{{
    size_t gid = get_global_id(0);
    {ph.axes_idxs_from_var('i', input_shape.rank, 'gid')}
    {';'.join([ f'i{i} += i{i}_offset' for i in range(input_shape.rank) ]) };
    dI[gid] += dO[O_idx({ph.axes_seq_enum('i', input_shape.rank)})];
}}
""")

def tile_test():
    for _ in range(10):
        for shape_len in range(3, 5):
            try:
                shape = tuple(np.random.randint( 8, size=(shape_len,) )+1)
                tiles = tuple(np.random.randint( 4, size=(shape_len,) )+1)

                val_n = np.random.randint( 2**8, size=shape ).astype(np.float32)
                tiled_n = np.tile(val_n, tiles)

                val_t = nn.Tensor_from_value(val_n)
                tiled_t = nn.tile(val_t, tiles)

                if tiled_n.shape != tiled_t.shape:
                    raise Exception(f'shape is not equal')

                if not all ( np.ndarray.flatten( tiled_t.np() == tiled_n ) ):
                    raise Exception(f'data is not equal')

                tiled_n_grad = np.random.randint( 2**8, size=tiled_n.shape ).astype(np.float32)

                val_t.get_grad().fill(1.0)
                nn.backward( {tiled_t:tiled_n_grad} , grad_for_non_trainables=True )

                info = nc.info.InfoTile( nc.TensorShape(shape), tiles)
                val_n_grad = sum([ tiled_n_grad[axes_slice] for axes_slice in info.axes_slices ])
                if not all ( np.ndarray.flatten(val_t.get_grad().np()-1.0 == val_n_grad) ):
                    raise Exception(f'grad is not equal')

            except:
                
                raise Exception(f"""
shape         : {shape}
tiles         : {tiles}
tiled_n_shape : {tiled_n.shape}
tiled_t_shape : {tiled_t.shape}
exception     : {traceback.format_exc()}
""")