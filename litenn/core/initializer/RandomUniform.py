import numpy as np
import litenn.core as nc
from litenn.core import CLKernelHelper as ph
from .Initializer import Initializer

class _RandomUniform(Initializer):
        
    def __init__(self, low=0.0, high=1.0):
        super().__init__()
        self.low = low
        self.high = high
        self.krn = nc.CLKernel(kernel_text=f"""
{ph.include_hash()}

__kernel void impl(__global float* O, uint seed)
{{
    size_t gid = get_global_id(0);
    O[gid] = hash_float_uint(gid+seed)*({high}-({low}))+({low});
}}
""")
        


    def initialize_CLBuffer(self, cl_buffer, tensor_shape):
        #value = np.random.uniform(self.low, self.high, size=(device_buffer.len,) )        
        #device_buffer.set(value)
        seed = np.random.randint(2147483648)
        cl_buffer.device.run( self.krn, cl_buffer, np.uint32(seed), global_shape=(cl_buffer.size // 4,) )
        
    def __str__(self):  return f'RandomUniform low={self.low}, high={self.high}'

def RandomUniform(low=0.0, high=1.0):
    """
    Initialize with random uniform.
    
    arguments
     
     low(0.0)   low value
     
     high(1.0)  high value
    """
    return nc.Cacheton.get(_RandomUniform, low, high)
