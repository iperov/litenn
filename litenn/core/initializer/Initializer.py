import litenn as nn

class Initializer:
    """
    Base class for inititalizers  
    """
    
    def has_fan_in_out(self):
        """
        Initializer has fan_in/fan_out parameters
        
         .fan_in .fan_out 
         if they are None, 
         they will be set by Conv2D, Dense and other layers
        """
        return False
        
    def initialize_CLBuffer(self, cl_buffer, tensor_shape):
        """
        Implement initialization of CLBuffer
        
        You can compute the data using python and then call cl_buffer.set(numpy_value)
        or call CLDevice.run CLKernel with cl_buffer to initialize the data using OpenCL        
        """
        raise NotImplementedError()
        
    def __str__(self): return 'Initializer'        
    def __repr__(self): return self.__str__()
