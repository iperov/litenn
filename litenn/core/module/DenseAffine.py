import numpy as np
import litenn as nn
import litenn.core as nc

class DenseAffine(nn.Dense):
    """
    Same as Dense, but
    
     1) outputs 6 element affine matrix Tensor (N,2,3), 
        that can be used in spatial_affine_transform2D
        
     2) initialized to produce identity affine matrix
    """
    
    
    def __init__(self, in_ch):
        super().__init__(in_ch, 6, weight_initializer=nn.initializer.Scalar(0.0), 
                                   bias_initializer=nn.initializer.NumpyArray(np.array([1,0,0,0,1,0])) )
        
    def forward(self, *args, **kwargs):
        x = super().forward(*args, **kwargs)
        return x.reshape( (-1,2,3) )
        
def DenseAffine_test():
    module = DenseAffine(4)        
    x = nn.Tensor( (2,4) )
    y = module(x)
    y.backward(grad_for_non_trainables=True)