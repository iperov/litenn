from .CLDevicesBuffers import CLDevicesBuffers
from .devices import devices

class CLKernel:
    """
    OpenCL kernel.

    It does not allocate any resources, thus can be used as static class variable.

    arguments

        kernel_text    OpenCL text of kernel. Must contain only one kernel.

        global_shape    default global_shape for .run()
        
        local_shape     default local_shape for .run()
    """
    def __init__(self, kernel_text, global_shape=None, local_shape=None):
        self.kernel_text = kernel_text
        self.global_shape = global_shape
        self.local_shape = local_shape
        
    def run(self, *args, global_shape=None, local_shape=None, global_shape_offsets=None):
        """
        Run kernel on all current devices

        Arguments
        
            *args           arguments which will be passed to OpenCL kernel
                            allowed types:
                            CLDevicesBuffers
                            np.int32 np.int64 np.uint32 np.uint64 np.float32

            global_shape(None)  tuple of ints, up to 3 dims
                                amount of parallel kernel executions.
                                in OpenCL kernel,
                                id can be obtained via get_global_id(dim)

            local_shape(None)   tuple of ints, up to 3 dims
                                specifies local groups of every dim of global_shape.
                                in OpenCL kernel,
                                id can be obtained via get_local_id(dim)

            global_shape_offsets(None)  tuple of ints
                                        offsets for global_shape

        Remark.
        
        if your op has different kernels for GPU and CPU devices, you should run them for every device manually.
        """

        if global_shape is None:
            global_shape = self.global_shape
            
        if local_shape is None:
            local_shape = self.local_shape

        for i, device in enumerate(devices.get_current()):
            device.run (self, *[arg._get_clbuffers()[i] if isinstance(arg, CLDevicesBuffers) else arg for arg in args],
                                global_shape=global_shape, 
                                local_shape=local_shape,
                                global_shape_offsets=global_shape_offsets)



    def __str__(self):  return f'CLKernel ({self.kernel_text})'
    def __repr__(self): return self.__str__()