import numpy as np
import litenn.core as nc
from litenn.core import CLKernelHelper as ph
from .Initializer import Initializer


        
        

class _MeshGrid2D(Initializer):
    krn = nc.CLKernel(kernel_text=f"""
__kernel void impl(__global float* O, float h_start, float h_step
                                    , float w_start, float w_step,
                                     uint OH, uint OW, uint OD)
{{
    size_t gid = get_global_id(0);
    
    size_t od = gid % OD; gid /= OD;
    size_t ow = gid % OW; gid /= OW;
    size_t oh = gid % OH;
    size_t oc = gid / OH; gid = get_global_id(0);
    
    float v = 1.0;
    if (od == 0)
        v = h_start+oh*h_step;
    else 
    if (od == 1)
        v = w_start+ow*w_step;
        
        
    O[gid] = v;
}}
""")

    def __init__(self, h_start, h_stop, w_start, w_stop):
        super().__init__()
        self.h_start = h_start
        self.h_stop = h_stop
        self.w_start = w_start
        self.w_stop = w_stop
        
        

    def initialize_CLBuffer(self, cl_buffer, tensor_shape : nc.TensorShape):
        if tensor_shape.rank != 4:
            raise ValueError(f'tensor_shape.rank must == 4')
        
        OC,OH,OW,OD = tensor_shape
        if OD != 3:
            raise ValueError(f'D {OD} must == 3')

        if OH > 1:
            h_step = (self.h_stop-self.h_start) / (OH-1)            
        else:
            h_step = 0
            
        if OW > 1:
            w_step = (self.w_stop-self.w_start) / (OW-1)            
        else:
            w_step = 0    
            
        cl_buffer.device.run(_MeshGrid2D.krn, cl_buffer, 
                             np.float32(self.h_start), np.float32(h_step),
                             np.float32(self.w_start), np.float32(w_step),
                             np.uint32(OH), np.uint32(OW), np.uint32(OD),
                             global_shape=(OC*OH*OW*OD,) )
        
    def __str__(self):  return f'MeshGrid2D'

def MeshGrid2D( h_start, h_stop, w_start, w_stop):
    """
    Initializes NCHW tensor with 2D mesh grid.
    
    arguments
        
     h_start
     h_stop
     w_start
     w_stop

    """
    return nc.Cacheton.get(_MeshGrid2D, h_start, h_stop, w_start, w_stop)
