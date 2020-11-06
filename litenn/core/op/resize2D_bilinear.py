from collections import Iterable
import numpy as np
import litenn as nn
import litenn.core as nc

def resize2D_bilinear(input_t, size_or_output_hw):
    """
    resize2D_bilinear operator
    
    arguments
    
     size_or_output_hw  int
                        float     
                        Iterable of height,weight
    
    """
    N,C,H,W = input_t.shape
    
    if isinstance(size_or_output_hw, Iterable):
        OH, OW = int(size_or_output_hw[0]), int(size_or_output_hw[1])
    elif isinstance(size_or_output_hw, (int, float)):
        OH = int(H * size_or_output_hw)
        OW = int(W * size_or_output_hw)
    else:
        raise ValueError(f'Unknown type of size_or_output_hw : {size_or_output_hw.__class__.__name__}')
    
    OH = max(1, OH)
    OW = max(1, OW)
    
    coords_shape = nc.TensorShape( (OH,OW,2) )
    
    coords_t = nn.Tensor( coords_shape, nn.initializer.CoordsArange(0, H-1, 0, W-1) )
    output_t = nn.spatial_transform2D(input_t, coords_t, grad_to_coords=False)

    return output_t
    
def resize2D_bilinear_test():
    for n in [1,4]:
        for ic in [1,2,4]:
            for iw,ih in zip(*[[4,8,16]]*2):
              for size in [0.6,1.0,2.0]:
                    try:
                        input_shape  = (n, ic, ih, iw)
                        input_n  = np.random.randint( 2**4, size=input_shape ).astype(np.float32)
                        
                        input_t  = nn.Tensor_from_value(input_n)

                        resized_t = nn.resize2D_bilinear(input_t, size)
 
                        nn.backward([resized_t], grad_for_non_trainables=True )
                    except:
                        raise Exception(f"""
input_shape    : {input_shape}
size           : {size}
resized_t.shape : {resized_t.shape}
{traceback.format_exc()}
""")
