import numpy as np
from .CLBuffer import CLBuffer
from .api import OpenCL as CL
from .CLShallowMode import CLShallowMode

class CLDevice:
    """
    Represents OpenCL Device

    available properties:

     .index     int     device index.

     .name      str

     .global_mem_size   int

     .is_cpu    bool

    """

    def __init__(self, device, index, name, global_mem_size, is_cpu):
        self.device = device
        self.index = index
        self.name = name
        self.global_mem_size = global_mem_size
        self.is_cpu = is_cpu

        self._buffers_pool = {}   # Pool of cached device buffers.
        self._cached_kernels = {} # Cached kernels
        self._ctx_q = None
        self._ctx = None

        self._total_buffers_allocated = 0
        self._total_buffers_pooled = 0
        self._total_memory_allocated = 0
        self._total_memory_pooled = 0

    def get_used_memory(self):
        """
        Returns amount of used memory
        """
        return self.get_total_allocated_memory() - self.get_total_pooled_memory()
        
    def get_total_allocated_memory(self):
        return self._total_memory_allocated

    def get_total_pooled_memory(self):
        return self._total_memory_pooled

    def wait(self):
        """
        Wait to finish all queued operations on this Device
        """
        CL.Finish(self._get_ctx_q())

    def run(self, kernel, *args, global_shape=None, local_shape=None, global_shape_offsets=None):
        """
        Run kernel on this Device

        Arguments

            *args           arguments will be passed to OpenCL kernel
                            allowed types:
                            CLBuffer
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


        """

        if CLShallowMode.stack_count > 0:
            return

        krn = self._cached_kernels.get(kernel, None)
        if krn is None:
            # Build kernel on the fly
            krn = CL.CreateProgramWithSource(self._get_ctx(), kernel.kernel_text)

            try:
                CL.BuildProgram(krn, [self.device], options="-cl-std=CL1.2 -cl-single-precision-constant")
            except CL.Exception as e:
                raise Exception(f'Build kernel fail: {CL.GetProgramBuildInfo(krn, CL.PROGRAM_BUILD_INFO.BUILD_LOG, self.device)}')

            kernels = CL.CreateKernelsInProgram(krn)
            if len(kernels) != 1:
                raise ValueError('CLKernel must contain only one __kernel.')
            krn = self._cached_kernels[kernel] = kernels[0]

        krn(self._get_ctx_q(), global_shape, local_shape, global_shape_offsets,
            *[arg.get_cl_mem() if isinstance(arg, CLBuffer) else arg for arg in args])

    ### INTERNAL METHODS start with _

    def _cleanup(self):
        """
        Frees all resources from this CLDevice.
        """
        if self._total_buffers_pooled != self._total_buffers_allocated:
            raise Exception('Unable to cleanup CLDevice, while not all buffers are pooled/deallocated.')

        # Lose reference is enough to free OpenCL resources.
        self._buffers_pool = {}
        self._cached_kernels = {}
        self._ctx_q = None
        self._ctx = None

    def _mem_alloc(self, size):
        """
        allocate or get cl_mem from pool
        """
        pool = self._buffers_pool

        # First try to get pooled buffer
        ar = pool.get(size, None)
        if ar is not None and len(ar) != 0:
            cl_mem = ar.pop(-1)
            self._total_memory_pooled -= size
            self._total_buffers_pooled -= 1
        else:
            # No pooled buffer, try to allocate new
            while True:
                cl_mem = None
                additional_log = ""
                try:
                    cl_mem = CL.CreateBuffer(self._get_ctx(), size)
                    # Fill one byte to check memory availability
                    CL.EnqueueFillBuffer(self._get_ctx_q(), cl_mem, np.uint8(0), 1, 0)
                    self._total_buffers_allocated += 1
                    self._total_memory_allocated += size
                    break
                except CL.Exception as e:

                    if e.error.value == CL.ERROR.MEM_OBJECT_ALLOCATION_FAILURE:
                        # MemoryError. Finding largest pooled buffer to release
                        buf_to_release = None
                        for size_key in sorted(list(pool.keys()), reverse=True):
                            ar = pool[size_key]
                            if len(ar) != 0:
                                buf_to_release = ar.pop(-1)
                                break

                        if buf_to_release is not None:
                            # Release pooled buffer and try to allocate again
                            del buf_to_release
                            continue
                    else:
                        additional_log = f'Unhandled CL exception {e}'
                except Exception as e:
                    additional_log = f'Unhandled exception {e}'

                raise Exception(f"Unable to allocate {size // 1024**2}Mb on {str(self)}. additional_log: {additional_log}")
        return cl_mem

    def _mem_free(self, cl_mem, size):
        """
        Put cl_mem to pool for reuse in future.
        """
        pool = self._buffers_pool
        ar = pool.get(size, None)
        if ar is None:
            ar = pool[size] = []
        ar.append(cl_mem)

        self._total_memory_pooled += size
        self._total_buffers_pooled += 1

    def _get_ctx(self):
        # Create OpenCL context on demand
        if self._ctx is None:
            self._ctx = CL.CreateContext(devices=[self.device])
        return self._ctx

    def _get_ctx_q(self):
        # Create CommandQueue on demand
        if self._ctx_q is None:
            self._ctx_q = CL.CreateCommandQueue(self._get_ctx(), self.device)
        return self._ctx_q

    def __repr__(self): return self.__str__()
    def __str__(self):  return f"[{self.index}]:[{self.name}][{self.global_mem_size / 1024**3 :.3} GB]"

    _shallow_mode_stack = 0