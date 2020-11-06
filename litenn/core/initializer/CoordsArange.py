import numpy as np
import litenn as nn
import litenn.core as nc
from .Initializer import Initializer

class _CoordsArange(Initializer):
    krn = nc.CLKernel(kernel_text=f"""
__kernel void impl(__global float* O, float h_start, float h_step
                                    , float w_start, float w_step,
                                     uint OH, uint OW, uint OD)
{{
    size_t gid = get_global_id(0);

    size_t od = gid % OD; gid /= OD;
    size_t ow = gid % OW; gid /= OW;
    size_t oh = gid % OH; gid = get_global_id(0);

    float v = 1.0;
    if (od == 0)
        v = w_start+ow*w_step;
    else
    if (od == 1)
        v = h_start+oh*h_step;

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
        if tensor_shape.rank not in [3,4,5]:
            raise ValueError(f'tensor_shape.rank must == 3,4,5')

        ON,OC = 1,1

        if tensor_shape.rank == 5:
            ON,OC,OH,OW,OD = tensor_shape
        elif tensor_shape.rank == 4:
            OC,OH,OW,OD = tensor_shape
        elif tensor_shape.rank == 3:
            OH,OW,OD = tensor_shape

        if OD not in [2,3]:
            raise ValueError(f'D {OD} must == 2 or 3')

        if OH > 1:
            h_step = (self.h_stop-self.h_start) / (OH-1)
        else:
            h_step = 0

        if OW > 1:
            w_step = (self.w_stop-self.w_start) / (OW-1)
        else:
            w_step = 0


        cl_buffer.device.run(_CoordsArange.krn, cl_buffer,
                             np.float32(self.h_start), np.float32(h_step),
                             np.float32(self.w_start), np.float32(w_step),
                             np.uint32(OH), np.uint32(OW), np.uint32(OD),
                             global_shape=(ON*OC*OH*OW*OD,) )

    def __str__(self):  return f'_CoordsArange'
    
def CoordsArange(h_start, h_stop, w_start, w_stop):
    """
    Initialize NCHWD, CHWD, HWD tensor with coords arange
     D == 2 or 3 (x,y,1)
     
    arguments
    
        h_start     float     height start value
        
        h_stop      float     height stop value
        
        w_start     float     width start value
        
        w_stop      float     width stop value
     
    """
    return nc.Cacheton.get(_CoordsArange, h_start, h_stop, w_start, w_stop)
