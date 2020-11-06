import traceback
import math
import numpy as np

import litenn as nn
import litenn.core as nc
from litenn.core import CLKernelHelper as ph

def resize2D_nearest(input_t, size=2):
    """
    resize2D_nearest operator
    
    arguments
    
        size(2)     int                               
    """
    return resize2D_nearest_op(input_t, size)
    
def resize2D_nearest_op (input_t, size=2, output_t=None, is_add_to_output=False):
    """
    arguments
    
        size(2)     int                         
    """
    is_add_to_output = False if output_t is None else is_add_to_output
    op = nc.Cacheton.get(_Resize2DNearestOp, input_t.shape, int(size), is_add_to_output)

    output_t = nn.Tensor( op.output_shape )
    output_t._set_op_name(f'resize2D_nearest')
    output_t._assign_gradfn (input_t, lambda O_t, dO_t: resize2D_nearest_dI_gradfn(op, input_t, dO_t) )

    op.O_forward_krn.run(output_t, input_t)
    return output_t

def resize2D_nearest_dI_gradfn(op, input_t, dO_t):
    op.dI_krn.run(input_t.get_grad(), dO_t)
    
class _Resize2DNearestOp:
    def __init__(self, input_shape : nc.TensorShape, size, is_add_to_output):                
        N,IC,IH,IW = input_shape 
        OC = IC
        OH = IH * size
        OW = IW * size
        
        self.output_shape = output_shape = nc.TensorShape( (N, OC, OH, OW) )

        common_kernel_text = f"""
{ph.define_axes_accessor('I', input_shape, 'NCHW')}
{ph.define_axes_accessor('O', output_shape, 'NCHW')}
"""
        
        
        self.O_forward_krn = nc.CLKernel(global_shape=(output_shape.size,), kernel_text=f"""
{common_kernel_text}

#define SIZE {size}

__kernel void impl(__global float* O, __global const float* I)
{{
    size_t gid = get_global_id(0);
    {ph.axes_idxs_from_var('o', 'NCHW', 'gid')}

    O[gid] {'+=' if is_add_to_output else '='} I[I_idx(on,oc,oh / SIZE,ow / SIZE)];
}}
""")
        self.dI_krn = nc.CLKernel(global_shape=(input_shape.size,), kernel_text=f"""
{common_kernel_text}

#define SIZE {size}

__kernel void impl(__global float* dI, __global const float* dO)
{{
    size_t gid = get_global_id(0);
    {ph.axes_idxs_from_var('i', 'NCHW', 'gid')}
    
    float v = 0.0;
    for (int y=0; y<SIZE; ++y)
    for (int x=0; x<SIZE; ++x)    
        v += dO[O_idx(in,ic,ih*SIZE+y,iw*SIZE+x)];
    
    dI[gid] += v;
}}
""")

def resize2D_nearest_test():
    for n in [1,4]:
        for ic in [1,2,4]:
            for iw,ih in zip(*[[4,8,16]]*2):
              for size in [2,3,4]:
                    try:
                        input_shape  = (n, ic, ih, iw)
                        input_n  = np.random.randint( 2**4, size=input_shape ).astype(np.float32)
                        
                        input_t  = nn.Tensor_from_value(input_n)

                        upsampled_t = nn.resize2D_nearest(input_t, size=size)

                        upsampled_n = input_n.reshape( (n, ic, ih, 1, iw, 1) )                        
                        upsampled_n = np.tile(upsampled_n, (1,1,1,size,1,size) )
                        upsampled_n = upsampled_n.reshape( (n,ic,ih*size, iw*size))
                        
                        if upsampled_n.shape != upsampled_t.shape:
                            raise Exception(f'shape is not equal')

                        if not all ( np.ndarray.flatten( upsampled_t.np() == upsampled_n) ):
                            raise Exception(f'data is not equal')
                        
                        upsampled_n_grad = np.random.randint( 2**8, size=upsampled_n.shape ).astype(np.float32)
                        
                        input_t.get_grad().fill(1.0)
                        nn.backward({ upsampled_t : upsampled_n_grad}, grad_for_non_trainables=True )
                        

                        input_n_grad = upsampled_n_grad.reshape( (n,ic,ih,size,iw,size) )
                        input_n_grad = input_n_grad.sum( (-3,-1) )
                        if not all ( np.ndarray.flatten( input_t.get_grad().np()-1.0) == np.ndarray.flatten(input_n_grad) ):
                            raise Exception('grad is not equal')
                    except:
                        raise Exception(f"""
input_shape    : {input_shape}
size           : {size}

upsampled_n.shape : {upsampled_n.shape}
upsampled_t.shape : {upsampled_t.shape}
{traceback.format_exc()}
""")
