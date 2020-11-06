import numpy as np
import litenn.core as nc
from litenn.core import CLKernelHelper as ph
from .Initializer import Initializer

class _RandomNormal(Initializer):
        
    def __init__(self, mean=0.0, stddev=1.0):
        super().__init__()
        self.krn = nc.CLKernel(kernel_text=f"""
{ph.include_constants_pi()}
{ph.include_hash()}

__kernel void impl(__global float* O, uint seed)
{{
    size_t gid = get_global_id(0);
    
    float2 rnd = hash_float2_uint(gid+seed);
    
    float rnd_normal = sqrt(-2*log(rnd.x))*cos(2*PI_F*rnd.y);
    
    O[gid] = {mean} + rnd_normal*{stddev};
}}
""")
        


    def initialize_CLBuffer(self, cl_buffer, tensor_shape):
        seed = np.random.randint(2147483648)
        cl_buffer.device.run( self.krn, cl_buffer, np.uint32(seed), global_shape=(cl_buffer.size // 4,) )
        
    def __str__(self):  return f'RandomNormal mean={self.mean}, stddev={self.stddev}'

def RandomNormal(mean=0.0, stddev=1.0):
    """
    Initialize with random normal.
    
    arguments
     
     mean(0.0)    mean value
     
     stddev(1.0)  stddev value
    """
    return nc.Cacheton.get(_RandomNormal, mean, stddev)
