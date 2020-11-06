from .devices import devices
from .CLBuffer import CLBuffer

class CLDevicesBuffers: 
    """
    Base class for per-device CLBuffer container for every current device.

    Can be argument for CLKernel.run()
    
    arguments
    
     size       Integer     size of buffer
     
     init_func(None)        function that receives (buffer) argument
                            called when memory is phisically allocated
                            and should be initialized
    """
    
    def __init__(self, size, init_func=None):
        self._clbuffers = [ # CLBuffer allocates physical mem on demand
              CLBuffer(device, size, init_func)
              for i, device in enumerate(devices.get_current()) ]
        
    def _get_clbuffers(self): return self._clbuffers