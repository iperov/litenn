import numpy as np
import litenn as nn
import litenn.core as nc
from litenn.core import CLKernelHelper as ph

def dropout(input_t, rate, seed=None):
    """
    Dropout operator
    
    arguments
    
        input_t     Tensor
        
        rate        float [0 .. 1.0)
                    probability
                    
        seed(None)  int value
                    if None - random seed
                    
    reference
    
    Dropout: A Simple Way to Prevent Neural Networks from Overfitting
    """
    return dropout_op(input_t, rate, seed)
    
def dropout_op(input_t, rate, seed=None, output_t=None, is_add_to_output=False):
    """
    Dropout operator
    
    arguments
    
        input_t     Tensor
        
        rate        float [0 .. 1.0)
                    probability
                    
        seed(None)  int value
                    if None - random seed
    
        output_t            compute result to this Tensor.
                            Tensor may be with different shape, but should match total size.
                            gradfn will not be set.

        is_add_to_output    add result to output_t if output_t is set.
        
    reference
    
    Dropout: A Simple Way to Prevent Neural Networks from Overfitting
    """

    op = nc.Cacheton.get(_DropoutOp, input_t.shape, float(rate), int(seed) if seed is not None else None, is_add_to_output)
    if output_t is None:
        output_t = nn.Tensor (op.output_shape)
        output_t._set_op_name('dropout')
        output_t._assign_gradfn (input_t, lambda O_t, dO_t: dropout_dI_gradfn(op, input_t, dO_t) )
    elif output_t.shape.size != op.output_shape.size:
        raise ValueError(f'output_t must have size {op.output_shape.size}')

    op.krn.run(output_t, input_t, np.uint32(op.seed)  )
    return output_t

def dropout_dI_gradfn(op, input_t, dO_t):
    dropout_op(dO_t, op.rate, op.seed, output_t=input_t.get_grad(), is_add_to_output=True )    
    
class _DropoutOp:
    def __init__(self, input_shape : nc.TensorShape, rate, seed, is_add_to_output):
        if rate < 0 or rate >= 1.0:
            raise ValueError(f'rate must be in range [0 .. 1.0)')

        self.rate = rate
        if seed is None:
            seed = np.random.randint(2147483648)
        self.seed = seed
        
        self.output_shape = input_shape
        
        self.krn = nc.CLKernel(global_shape=(input_shape.size,), kernel_text=f"""
{ph.include_hash()}

{ph.define_axes_accessor('I', input_shape)}
{ph.define_axes_accessor('O', self.output_shape)}

__kernel void impl(__global float* O, __global const float* I, uint seed)
{{
    size_t gid = get_global_id(0);

    float value = 0.0;
    if ( hash_float_uint(seed+gid) <= {rate} )
        value = I[gid] * ( 1.0 / ( 1.0 - {rate} ) );
        
    O[gid] {'+=' if is_add_to_output else '='} value;
}}""")

def dropout_test():
    for _ in range(10):
        for shape_len in range(2, 5):
            try:
                shape = np.random.randint( 1, 8, size=(shape_len,) )
                rate = np.random.uniform()
                val_n = np.random.randint( 1, 2**8, size=shape ).astype(np.float32)
                
                val_t = nn.Tensor_from_value(val_n)
                result_t = dropout(val_t, rate)
                result_t.backward(grad_for_non_trainables=True)
                
                if not all ( np.ndarray.flatten( np.argwhere( result_t.np() == 0 ) == \
                                                 np.argwhere( val_t.get_grad().np() == 0 ) )):
                    raise Exception(f'dI is wrong.')
                
            except:
                raise Exception(f"""
shape              : {shape}
exception          : {traceback.format_exc()}
""")
