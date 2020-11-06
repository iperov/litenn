import numpy as np
import litenn.core as nc
from .Initializer import Initializer

class _ScalarInit(Initializer):
    def __init__(self, value):
        super().__init__()
        self.value = value
        
        
    def initialize_CLBuffer(self, cl_buffer : nc.CLBuffer, tensor_shape):
        cl_buffer.fill( np.float32(self.value) )
        
    def __str__(self):  return f'Scalar ({self.value})'

def Scalar(value):
    """
    Initialize with scalar value.
    """
    return nc.Cacheton.get(_ScalarInit, value)
