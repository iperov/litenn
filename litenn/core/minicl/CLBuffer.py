import numpy as np
from .api import OpenCL as CL
from .CLShallowMode import CLShallowMode

class CLBuffer:
    """
    Represents device memory buffer.

    Physical allocation is performed only on get_cl_mem() access.

     size       Integer     size of buffer in bytes

     init_func(None)        function that receives (CLBuffer) argument
                            called when memory is phisically allocated
                            and should be initialized
    """

    __slots__ = ['device','size','init_func','_cl_mem']

    def __init__(self, device, size, init_func=None ):
        CLBuffer._object_count += 1
        self._cl_mem = None

        if size <= 0:
            raise ValueError(f'Create CLBuffer with {size} size.')

        self.device = device
        self.size = size
        self.init_func = init_func
        
    def __del__(self):
        CLBuffer._object_count -= 1
        self.free_cl_mem()    

    def has_cl_mem(self): return self._cl_mem is not None
    def get_cl_mem(self):
        if self._cl_mem is None:
            self._cl_mem = self.device._mem_alloc(self.size)
            if self.init_func is not None:
                self.init_func(self)

        return self._cl_mem

    def free_cl_mem(self):
        if self._cl_mem is not None:
            self.device._mem_free(self._cl_mem, self.size)
            self._cl_mem = None


    def set(self, value):
        """
        Parameters

            value   CLBuffer    copy data from other CLBuffer.
            
                    np.ndarray  copies values from ndarray 
                                to CLBuffer's memory

        """            
        if isinstance(value, CLBuffer):
            if self != value:
                if self.size != value.size:
                    raise Exception(f'Unable to copy from CLBuffer with {value.size} size to buffer with {self.size} size.')

                if CLShallowMode.stack_count == 0:
                    if self.device == value.device:
                        CL.EnqueueCopyBuffer ( self.device._get_ctx_q(), value.get_cl_mem(), self.get_cl_mem(), 0,0, self.size )
                    else:
                        # Transfer between devices will cause slow performance
                        self.set( value.np() )
        else:
            if not isinstance(value, np.ndarray):
                raise ValueError (f'Invalid type {value.__class__}. Must be np.ndarray.')

            if value.nbytes != self.size:
                raise ValueError(f'Value size {value.nbytes} does not match CLBuffer size {self.size}.')

            if CLShallowMode.stack_count == 0:
                # wait upload, otherwise value's memory can be disappeared
                CL.ndarray_to_buffer ( self.device._get_ctx_q(), self.get_cl_mem(), value).wait()

    _allowed_fill_types = (np.float16, np.float32, np.float64, np.uint8, np.int8, np.int16, np.int32, np.int64)
    def fill(self, value):
        """
        Fills buffer with scalar value.

        arguments

            value   np.float16, np.float32, np.float64, np.uint8, np.int8, np.int16, np.int32, np.int64
        """
      
        if not isinstance(value, CLBuffer._allowed_fill_types ):
            raise ValueError(f'Unknown type {value.__class__}. Allowed types : {CLBuffer._allowed_fill_types } ')
        
        if CLShallowMode.stack_count == 0:
            CL.EnqueueFillBuffer (self.device._get_ctx_q(), self.get_cl_mem(), value, self.size)

    _dtype_size_dict = { np.float32 : 4}
    def np(self, shape=None, dtype=np.float32):
        """
        Returns data of buffer as np.ndarray with specified shape and dtype
        """   
        dtype_size = CLBuffer._dtype_size_dict[dtype]
        
        if shape is None or len(shape) == 0:
            shape = (self.size // dtype_size,)
        out_np_value = np.empty ( shape, dtype )
        
        if out_np_value.nbytes != self.size:
            raise ValueError(f'Unable to represent CLBuffer with size {self.size} as shape {shape} with dtype {dtype}')        

        if CLShallowMode.stack_count == 0:
            CL.buffer_to_ndarray( self.device._get_ctx_q(), self.get_cl_mem(), out_np_value ).wait()
            
        return out_np_value


    def __str__(self):
        return f'CLBuffer [{self.size} bytes] on {str(self.device)}'

    def __repr__(self):
        return self.__str__()

    _object_count = 0