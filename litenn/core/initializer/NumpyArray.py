import numpy as np
import litenn.core as nc
from .Initializer import Initializer

class _NumpyArrayInit(Initializer):
    def __init__(self, np_array):
        super().__init__()
        self.np_array = np_array.astype(np.float32).copy()
        
        
    def initialize_CLBuffer(self, cl_buffer : nc.CLBuffer, tensor_shape):
        cl_buffer.set(self.np_array)
        
    def __str__(self):  return f'NumpyArray {self.np_array.shape}'

def NumpyArray(np_array):
    """
    Initialize with np.ndarray.
    The value will copied to internal structure.
    """
    return _NumpyArrayInit(np_array)
