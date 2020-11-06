import traceback
import numpy as np
import litenn as nn
import litenn.core as nc
from litenn.core import CLKernelHelper as ph

def reshape(input_t, target_shape):
    """
    reshape operator

    arguments

        target_shape    Iterable of ints
        
    Produces reference Tensor with new shape.                        
    """    
    info = nc.Cacheton.get(nc.info.InfoReshape, input_t.shape, tuple(int(x) for x in target_shape) )
    return input_t._as_ref( info.output_shape )

def flatten(input_t):
    """
    same as 
     
     nn.reshape(x, (x.shape[0],-1) )
    """
    return reshape(input_t, (input_t.shape[0],-1) )
    