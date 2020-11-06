"""
Minimal OpenCL 1.2 API
"""
import ctypes
import os
import sys
from ctypes import POINTER, byref
from ctypes import c_char_p as char_p
from ctypes import c_size_t as size_t
from ctypes import c_void_p as void_p
from ctypes import cast, create_string_buffer, sizeof
from typing import List

import numpy as np

lib = None
try:
    from ctypes.util import find_library as _find_library
    lib = ctypes.cdll.LoadLibrary(_find_library('OpenCL'))
except:
    pass
if lib is None:
    raise RuntimeError('Could not load OpenCL libary.')

class cl_char(ctypes.c_int8): pass
class cl_uchar(ctypes.c_uint8): pass
class cl_short(ctypes.c_int16): pass
class cl_ushort(ctypes.c_uint16): pass
class cl_int(ctypes.c_int32): pass
class cl_uint(ctypes.c_uint32): pass
class cl_long(ctypes.c_int64): pass
class cl_ulong(ctypes.c_uint64): pass
class cl_half(ctypes.c_uint16): pass
class cl_float(ctypes.c_float): pass
class cl_double(ctypes.c_double): pass
class cl_bool(cl_uint): pass

def get_enumname_by_value(cls, value):
    _ebv = getattr(cls, '_ebv', None)
    if _ebv is None:
        _ebv = {}
        for name, name_value in cls.__dict__.items():
            if isinstance(name_value, int):
                _ebv[name_value] = name
        cls._ebv = _ebv
    return _ebv.get(value, "")

class cl_uenum(cl_uint):
    def __eq__(self, other):
        if not isinstance(other, self.__class__):
            return False
        else:
            return self.value == other.value
    def __ne__(self, other):
        return not(self == other)
    def __hash__(self):
        return self.value.__hash__()
    def __str__(self):
        return get_enumname_by_value(self.__class__, self.value)
    def __repr__(self):
        return self.__str__()


class cl_enum(cl_int):
    def __eq__(self, other):
        if not isinstance(other, self.__class__):
            return False
        else:
            return self.value == other.value
    def __ne__(self, other):
        return not(self == other)
    def __hash__(self):
        return self.value.__hash__()

    def __str__(self):
        return get_enumname_by_value(self.__class__, self.value)
    def __repr__(self):
        return self.__str__()

class cl_bitfield(cl_ulong):
    def __or__(self, other):
        assert isinstance(other, self.__class__)
        return self.__class__(self.value | other.value)
    def __and__(self, other):
        assert isinstance(other, self.__class__)
        return self.__class__(self.value & other.value)
    def __xor__(self, other):
        assert isinstance(other, self.__class__)
        return self.__class__(self.value ^ other.value)
    def __not__(self):
        return self.__class__(~self.value)
    def __contains__(self, other):
        assert isinstance(other, self.__class__)
        return (self.value & other.value) == other.value
    def __hash__(self):
        return self.value.__hash__()
    def __eq__(self, other):
        if not isinstance(other, self.__class__):
            return False
        else:
            return self.value == other.value
    def __ne__(self, other):
        return not(self == other)
    def __repr__(self):
        by_value = self.__class__._by_value
        names = []
        if self in by_value:
            return by_value[self]
        for val in by_value:
            if val in self:
                names.append(by_value[val])
        if names:
            return " | ".join(names)
        elif self.value:
            return "UNKNOWN(0x%x)" % self.value
        else:
            return "NONE"

# Errors

class ERROR(cl_enum):
    SUCCESS =                                  0
    DEVICE_NOT_FOUND =                         -1
    DEVICE_NOT_AVAILABLE =                     -2
    COMPILER_NOT_AVAILABLE =                   -3
    MEM_OBJECT_ALLOCATION_FAILURE =            -4
    OUT_OF_RESOURCES =                         -5
    OUT_OF_HOST_MEMORY =                       -6
    PROFILING_INFO_NOT_AVAILABLE =             -7
    MEM_COPY_OVERLAP =                         -8
    IMAGE_FORMAT_MISMATCH =                    -9
    IMAGE_FORMAT_NOT_SUPPORTED =               -10
    BUILD_PROGRAM_FAILURE =                    -11
    MAP_FAILURE =                              -12
    MISALIGNED_SUB_BUFFER_OFFSET =             -13
    EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST = -14
    INVALID_VALUE =                            -30
    INVALID_DEVICE_TYPE =                      -31
    INVALID_PLATFORM =                         -32
    INVALID_DEVICE =                           -33
    INVALID_CONTEXT =                          -34
    INVALID_QUEUE_PROPERTIES =                 -35
    INVALID_COMMAND_QUEUE =                    -36
    INVALID_HOST_PTR =                         -37
    INVALID_MEM_OBJECT =                       -38
    INVALID_IMAGE_FORMAT_DESCRIPTOR =          -39
    INVALID_IMAGE_SIZE =                       -40
    INVALID_SAMPLER =                          -41
    INVALID_BINARY =                           -42
    INVALID_BUILD_OPTIONS =                    -43
    INVALID_PROGRAM =                          -44
    INVALID_PROGRAM_EXECUTABLE =               -45
    INVALID_KERNEL_NAME =                      -46
    INVALID_KERNEL_DEFINITION =                -47
    INVALID_KERNEL =                           -48
    INVALID_ARG_INDEX =                        -49
    INVALID_ARG_VALUE =                        -50
    INVALID_ARG_SIZE =                         -51
    INVALID_KERNEL_ARGS =                      -52
    INVALID_WORK_DIMENSION =                   -53
    INVALID_WORK_GROUP_SIZE =                  -54
    INVALID_WORK_ITEM_SIZE =                   -55
    INVALID_GLOBAL_OFFSET =                    -56
    INVALID_EVENT_WAIT_LIST =                  -57
    INVALID_EVENT =                            -58
    INVALID_OPERATION =                        -59
    INVALID_GL_OBJECT =                        -60
    INVALID_BUFFER_SIZE =                      -61
    INVALID_MIP_LEVEL =                        -62
    INVALID_GLOBAL_WORK_SIZE =                 -63
    INVALID_PROPERTY =                         -64
    INVALID_GL_SHAREGROUP_REFERENCE_KHR =      -1000
    PLATFORM_NOT_FOUND_KHR =                   -1001

class CLException(Exception):
    def __init__(self, error, message=None):
        self.error = error
        if message is None:
            message = ""
        self.message = message
    def __repr__(self):
        return self.__str__()
    def __str__(self):
        return f"{self.error} {self.message}"

def _errcheck_result(result, func, args):
    error_code = result.value

    if error_code != ERROR.SUCCESS:
        raise CLException( ERROR(error_code) )
    return result

def _errcheck_lastarg(result, func, args):
    lastarg = args[-1]
    if not hasattr(lastarg, '_obj'):
        raise Exception()
    _errcheck_result(lastarg._obj , func, args)
    return result

# Platforms

class PLATFORM_INFO(cl_uenum):
    PLATFORM_NAME = 0x0902

class platform(void_p):
    def __repr__(self):
        try:
            return "<platform '%s'>" % self.name
        except:
            return "<platform 0x%x>" % (self.value or 0)
    @property
    def name(self): return GetPlatformInfo(self, PLATFORM_INFO.PLATFORM_NAME)

lib_GetPlatformIDs = lib.clGetPlatformIDs
lib_GetPlatformIDs.argtypes = cl_uint, POINTER(platform), POINTER(cl_uint)
lib_GetPlatformIDs.restype  = ERROR
lib_GetPlatformIDs.errcheck = _errcheck_result
def GetPlatforms() -> List[platform]:
    num_platforms = cl_uint()
    lib_GetPlatformIDs(0, None, byref(num_platforms))
    n = num_platforms.value
    if n > 0:
        platform_array = (platform * n)()
        lib_GetPlatformIDs(num_platforms, platform_array, None)
        return tuple(x for x in platform_array)
    else:
        return ()

lib_GetPlatformInfo = lib.clGetPlatformInfo
lib_GetPlatformInfo.argtypes = platform, PLATFORM_INFO, size_t, void_p, POINTER(size_t)
lib_GetPlatformInfo.restype  = ERROR
lib_GetPlatformInfo.errcheck = _errcheck_result
def GetPlatformInfo(platform, param_name):
    sz = size_t()
    lib_GetPlatformInfo(platform, param_name, 0, None, byref(sz))
    param_value = create_string_buffer(sz.value)
    lib_GetPlatformInfo(platform, param_name, sz.value, param_value, None)
    return str(param_value.value, 'ascii')

# Devices

class DEVICE_TYPE(cl_bitfield):
    DEFAULT =     (1 << 0)
    CPU =         (1 << 1)
    GPU =         (1 << 2)
    ACCELERATOR = (1 << 3)
    ALL =         0xFFFFFFFF

class DEVICE_INFO(cl_uenum):
    MAX_MEM_ALLOC_SIZE = 0x1010
    GLOBAL_MEM_SIZE =    0x101F
    NAME =               0x102B
    VERSION =            0x102F

class device(void_p):
    @property
    def device_version(self):
        return GetDeviceInfo(self, DEVICE_INFO.VERSION)
    @property
    def global_mem_size(self):
        return GetDeviceInfo(self, DEVICE_INFO.GLOBAL_MEM_SIZE)
    @property
    def max_mem_alloc_size(self):
        return GetDeviceInfo(self, DEVICE_INFO.MAX_MEM_ALLOC_SIZE)
    @property
    def name(self):
        return GetDeviceInfo(self, DEVICE_INFO.NAME)
    def __repr__(self):
        try:
            return "<device '%s'>" % (self.name)
        except:
            return "<device 0x%x>" % (self.value or 0)

lib_GetDeviceIDs = lib.clGetDeviceIDs
lib_GetDeviceIDs.argtypes = platform, DEVICE_TYPE, cl_uint, POINTER(device), POINTER(cl_uint)
lib_GetDeviceIDs.restype  = ERROR
lib_GetDeviceIDs.errcheck = _errcheck_result
def GetDeviceIDs(platform, DEVICE_TYPE = DEVICE_TYPE.ALL) -> List[device]:
    num_devices = cl_uint()

    try:
        lib_GetDeviceIDs(platform, DEVICE_TYPE, 0, None, byref(num_devices))
    except CLException as clerr:
        if clerr.error.value == ERROR.DEVICE_NOT_FOUND:
            num_devices.value = 0
        else:
            raise

    n = num_devices.value
    if n > 0:
        device_array = (device*n)()
        lib_GetDeviceIDs(platform, DEVICE_TYPE, num_devices, device_array, None)
        return tuple(x for x in device_array)
    else:
        return ()

lib_GetDeviceInfo = lib.clGetDeviceInfo
lib_GetDeviceInfo.argtypes = device, DEVICE_INFO, size_t, void_p, POINTER(size_t)
lib_GetDeviceInfo.restype  = ERROR
lib_GetDeviceInfo.errcheck = _errcheck_result
def GetDeviceInfo(device, param_name):
    if param_name == DEVICE_INFO.MAX_MEM_ALLOC_SIZE or  \
       param_name == DEVICE_INFO.GLOBAL_MEM_SIZE:
        param_value = cl_ulong()
        lib_GetDeviceInfo(device, param_name, sizeof(param_value), byref(param_value), None)
        return int(param_value.value)
    elif param_name == DEVICE_INFO.NAME or \
         param_name == DEVICE_INFO.VERSION:
        sz = size_t()
        lib_GetDeviceInfo(device, param_name, 0, None, byref(sz))
        param_value = create_string_buffer(sz.value)
        lib_GetDeviceInfo(device, param_name, sz, param_value, None)
        return str(param_value.value, 'ascii')

# Context

class context(void_p):
    def __del__(self):
        if self.value is not None:
            ReleaseContext(self)

lib_CreateContext = lib.clCreateContext
lib_CreateContext.argtypes = void_p, cl_uint, POINTER(device), void_p, void_p, POINTER(ERROR)
lib_CreateContext.restype  = context
lib_CreateContext.errcheck = _errcheck_lastarg
def CreateContext(devices):
    device_array = (device * len(devices))()
    for i, d in enumerate(devices):
        device_array[i] = d
    return lib_CreateContext(None, len(devices), device_array, None, None, byref(ERROR()))

lib_ReleaseContext = lib.clReleaseContext
lib_ReleaseContext.argtypes = context,
lib_ReleaseContext.restype  = ERROR
lib_ReleaseContext.errcheck = _errcheck_result
def ReleaseContext(context): lib_ReleaseContext(context)

# Command Queue

class command_queue(void_p):
    def __del__(self):
        if self.value is not None:
            ReleaseCommandQueue(self)

lib_CreateCommandQueue = lib.clCreateCommandQueue
lib_CreateCommandQueue.argtypes = context, device, cl_uint, POINTER(ERROR)
lib_CreateCommandQueue.restype  = command_queue
lib_CreateCommandQueue.errcheck = _errcheck_lastarg
def CreateCommandQueue(context, device):
    queue = lib_CreateCommandQueue(context, device, cl_uint(0), byref(ERROR()))
    return queue

lib_Finish = lib.clFinish
lib_Finish.argtypes = command_queue,
lib_Finish.restype  = ERROR
lib_Finish.errcheck = _errcheck_result
def Finish(queue): lib_Finish(queue)


lib_ReleaseCommandQueue = lib.clReleaseCommandQueue
lib_ReleaseCommandQueue.argtypes = command_queue,
lib_ReleaseCommandQueue.restype  = ERROR
lib_ReleaseCommandQueue.errcheck = _errcheck_result
def ReleaseCommandQueue(queue): lib_ReleaseCommandQueue(queue)

# Events

class event(void_p):
    def wait(self):
        if self.value is not None:
            WaitForEvent(self)
    def __del__(self):
        if self.value is not None:
            ReleaseEvent(self)


lib_WaitForEvents = lib.clWaitForEvents
lib_WaitForEvents.argtypes = cl_uint, POINTER(event)
lib_WaitForEvents.restype  = ERROR
lib_WaitForEvents.errcheck = _errcheck_result
def WaitForEvent(ev):
    event_array = (event * 1)()
    event_array[0] = ev
    lib_WaitForEvents(1, event_array)

lib_ReleaseEvent = lib.clReleaseEvent
lib_ReleaseEvent.argtypes = event,
lib_ReleaseEvent.restype  = ERROR
lib_ReleaseEvent.errcheck = _errcheck_result
def ReleaseEvent(ev): lib_ReleaseEvent(ev)

# Memory objects

class MEM_FLAGS(cl_bitfield):
    MEM_READ_WRITE = (1 << 0)

class mem(void_p):
    def __del__(self):
        if self.value is not None:
            ReleaseMemObject(self)

lib_CreateBuffer = lib.clCreateBuffer
lib_CreateBuffer.argtypes = context, MEM_FLAGS, size_t, void_p, POINTER(ERROR)
lib_CreateBuffer.restype  = mem
lib_CreateBuffer.errcheck = _errcheck_lastarg
def CreateBuffer(context, size):
    return lib_CreateBuffer(context, MEM_FLAGS.MEM_READ_WRITE, size, None, byref(ERROR()))

lib_ReleaseMemObject = lib.clReleaseMemObject
lib_ReleaseMemObject.argtypes = mem,
lib_ReleaseMemObject.restype  = ERROR
lib_ReleaseMemObject.errcheck = _errcheck_result
def ReleaseMemObject(mem): lib_ReleaseMemObject(mem)

lib_EnqueueReadBuffer = lib.clEnqueueReadBuffer
lib_EnqueueReadBuffer.argtypes = command_queue, mem, cl_bool, size_t, size_t, void_p, cl_uint, POINTER(event), POINTER(event)
lib_EnqueueReadBuffer.restype  = ERROR
lib_EnqueueReadBuffer.errcheck = _errcheck_result
def EnqueueReadBuffer(queue, mem, pointer, size, offset=0):
    out_event = event()
    lib_EnqueueReadBuffer(queue, mem, 0, offset, size, pointer, 0, None, byref(out_event))
    return out_event

lib_EnqueueWriteBuffer = lib.clEnqueueWriteBuffer
lib_EnqueueWriteBuffer.argtypes = command_queue, mem, cl_bool, size_t, size_t, void_p, cl_uint, POINTER(event), POINTER(event)
lib_EnqueueWriteBuffer.restype  = ERROR
lib_EnqueueWriteBuffer.errcheck = _errcheck_result
def EnqueueWriteBuffer(queue, mem, pointer, size, offset=0):
    out_event = event()
    lib_EnqueueWriteBuffer(queue, mem, 0, offset, size, pointer, 0, None, byref(out_event))
    return out_event

lib_EnqueueCopyBuffer = lib.clEnqueueCopyBuffer
lib_EnqueueCopyBuffer.argtypes = command_queue, mem, mem, size_t, size_t, void_p, cl_uint, POINTER(event), POINTER(event)
lib_EnqueueCopyBuffer.restype  = ERROR
lib_EnqueueCopyBuffer.errcheck = _errcheck_result
def EnqueueCopyBuffer(queue, src_buffer, dst_buffer, src_offset, dst_offset, size):
    out_event = event()
    lib_EnqueueCopyBuffer(queue, src_buffer, dst_buffer, src_offset, dst_offset, size, 0, None, byref(out_event))
    return out_event

lib_EnqueueFillBuffer = lib.clEnqueueFillBuffer
lib_EnqueueFillBuffer.argtypes = command_queue, mem, void_p, size_t, size_t, void_p, cl_uint, POINTER(event), POINTER(event)
lib_EnqueueFillBuffer.restype  = ERROR
lib_EnqueueFillBuffer.errcheck = _errcheck_result
def EnqueueFillBuffer(queue, mem, pattern, size, offset=0):
    if isinstance(pattern, (np.float16, np.float32, np.float64, np.uint8, np.int8, np.int16, np.int32, np.int64)):
        pattern = np.array([pattern], dtype=pattern.dtype)
    if isinstance(pattern, np.ndarray):
        pattern_size = pattern.nbytes
        pattern = pattern.ctypes.data
    else:
        raise ValueError('unknown type of pattern')
    out_event = event()
    lib_EnqueueFillBuffer(queue, mem, pattern, pattern_size, offset, max(pattern_size,size), 0, None, byref(out_event))
    return out_event


def buffer_to_ndarray(queue, buf, np_ar):
    """
    Fill np.ndarray from buffer from.

    returns

        event of queued operation.

    You must wait() the event,
    if lifetime of np_ar is not gauranteed until the end of the operation.
    """
    if not np_ar.flags.contiguous:
        raise ValueError ("Unable to write to non-contiguous np array.")
    evt = EnqueueReadBuffer(queue, buf, np_ar.ctypes.data, np_ar.nbytes)
    return evt

def ndarray_to_buffer(queue, buf, np_ar):
    """
    Fill buffer from np.ndarray.

    returns

        event of queued operation.

    You must wait() the event,
    if lifetime of np_ar is not gauranteed until the end of the operation.
    """
    if not np_ar.flags.contiguous:
        np_ar = np_ar.reshape(-1)

    if not np_ar.flags.contiguous:
        raise ValueError ("Unable to write from non-contiguous np array.")

    evt = EnqueueWriteBuffer(queue, buf, np_ar.ctypes.data_as(void_p), np_ar.nbytes)
    return evt

# Program objects

class PROGRAM_INFO(cl_uenum):
    BINARY_SIZES =                     0x1165
    BINARIES =                         0x1166

class PROGRAM_BUILD_INFO(cl_uenum):
    STATUS    = 0x1181
    BUILD_LOG = 0x1183

class BUILD_STATUS(cl_enum):
    ERROR = -2

class program(void_p):
    def __del__(self):
        if self.value is not None:
            self.kernels = None
            ReleaseProgram(self)
            self.value = None

lib_ReleaseProgram = lib.clReleaseProgram
lib_ReleaseProgram.argtypes = program,
lib_ReleaseProgram.restype  = ERROR
lib_ReleaseProgram.errcheck = _errcheck_result
def ReleaseProgram(program): lib_ReleaseProgram(program)

lib_GetProgramInfo = lib.clGetProgramInfo
lib_GetProgramInfo.argtypes = program, PROGRAM_INFO, size_t, void_p, POINTER(size_t)
lib_GetProgramInfo.restype  = ERROR
lib_GetProgramInfo.errcheck = _errcheck_result
def GetProgramInfo(program, param_name):
    if param_name == PROGRAM_INFO.BINARY_SIZES:
        sz = size_t()
        lib_GetProgramInfo(program, param_name, 0, None, byref(sz))
        nd = sz.value // sizeof(size_t)
        param_value = (size_t * nd)()
        lib_GetProgramInfo(program, param_name, sz, param_value, None)
        return [int(x) for x in param_value]
    elif param_name == PROGRAM_INFO.BINARIES:
        sz = size_t()
        lib_GetProgramInfo(program, param_name, 0, None, byref(sz))
        nd = sz.value // sizeof(char_p)
        param_value = (char_p * nd)()
        binary_sizes = GetProgramInfo(program, PROGRAM_INFO.BINARY_SIZES)
        binaries = [None]*nd
        for i, binary_size in enumerate(binary_sizes):
            binaries[i] = (ctypes.c_char * binary_size)()
            param_value[i] = cast(binaries[i], char_p)
        lib_GetProgramInfo(program, param_name, sz, param_value, None)
        return [ str(x.value, encoding='ascii') for x in binaries]
    else:
        raise ValueError("Unknown program info %s" % param_name)

lib_CreateProgramWithSource = lib.clCreateProgramWithSource
lib_CreateProgramWithSource.argtypes = context, cl_uint, POINTER(char_p), POINTER(size_t), POINTER(ERROR)
lib_CreateProgramWithSource.restype  = program
lib_CreateProgramWithSource.errcheck = _errcheck_lastarg
def CreateProgramWithSource(context, source):
    source = source.encode()
    return lib_CreateProgramWithSource(context, 1, char_p(source), None, byref(ERROR()))

lib_GetProgramBuildInfo = lib.clGetProgramBuildInfo
lib_GetProgramBuildInfo.argtypes = program, device, PROGRAM_BUILD_INFO, size_t, void_p, POINTER(size_t)
lib_GetProgramBuildInfo.restype  = ERROR
lib_GetProgramBuildInfo.errcheck = _errcheck_result
def GetProgramBuildInfo(program, param_name, device):
    if param_name == PROGRAM_BUILD_INFO.STATUS:
        param_value = BUILD_STATUS()
        lib_GetProgramBuildInfo(program, device, param_name, sizeof(param_value), byref(param_value), None)
        return param_value
    elif param_name == PROGRAM_BUILD_INFO.BUILD_LOG:
        sz = size_t()
        lib_GetProgramBuildInfo(program, device, param_name, 0, None, byref(sz))
        param_value = create_string_buffer(sz.value)
        lib_GetProgramBuildInfo(program, device, param_name, sz, param_value, None)
        return str(param_value.value, 'utf-8')
    else:
        raise ValueError("Unknown program build info %s" % param_name)

lib_BuildProgram = lib.clBuildProgram
lib_BuildProgram.argtypes = program, cl_uint, POINTER(device), char_p, void_p, void_p
lib_BuildProgram.restype  = ERROR
lib_BuildProgram.errcheck = _errcheck_result
def BuildProgram(program, devices, options=None):
    if options is not None:
        options = char_p(options.encode('utf-8'))

    num_devices = len(devices)
    dev_array = (device*num_devices)()
    for i,dev in enumerate(devices):
        dev_array[i] = dev
    lib_BuildProgram(program, num_devices, dev_array, options, None, None)

# Kernel objects

class KERNEL_INFO(cl_uenum):
    FUNCTION_NAME = 0x1190
    NUM_ARGS =      0x1191

class kernel(void_p):
    dtype_np_to_cl = { np.int16: cl_short,
                       np.int32: cl_int,
                       np.int64: cl_long,
                       np.uint32 : cl_uint,
                       np.uint64 : cl_ulong,
                       np.float16: cl_half,
                       np.float32: cl_float,
                       np.float64: cl_double }

    def __call__(self, queue, global_shape, local_shape=None, shape_offsets=None, *args, **kw):
        for index, arg in enumerate(args):
            if isinstance(arg, mem):
                size = sizeof(mem)
            else:
                cl_type = kernel.dtype_np_to_cl.get( arg.__class__, None)
                if cl_type is None:
                    raise ValueError(f'Cannot convert type {arg.__class__} to OpenCL type.')
                arg = cl_type(arg)
                size = sizeof(arg)
            SetKernelArg(self, index, size, byref(arg))
        return EnqueueNDRangeKernel(queue, self, global_shape, local_shape, shape_offsets)

    def __del__(self):
        if self.value is not None:
            ReleaseKernel(self)

lib_CreateKernelsInProgram = lib.clCreateKernelsInProgram
lib_CreateKernelsInProgram.argtypes = program, cl_uint, POINTER(kernel), POINTER(cl_uint)
lib_CreateKernelsInProgram.restype  = ERROR
lib_CreateKernelsInProgram.errcheck = _errcheck_result
def CreateKernelsInProgram(program):
    n_kernels = cl_uint()
    lib_CreateKernelsInProgram(program, 0, None, byref(n_kernels))
    n_kernels = n_kernels.value
    p_kernels = ( kernel * n_kernels )()
    lib_CreateKernelsInProgram(program, n_kernels, p_kernels, None)
    return [ x for x in p_kernels]

lib_ReleaseKernel = lib.clReleaseKernel
lib_ReleaseKernel.argtypes = kernel,
lib_ReleaseKernel.restype  = ERROR
lib_ReleaseKernel.errcheck = _errcheck_result
def ReleaseKernel(kernel): lib_ReleaseKernel(kernel)

lib_SetKernelArg = lib.clSetKernelArg
lib_SetKernelArg.argtypes = kernel, cl_uint, size_t, void_p
lib_SetKernelArg.restype  = ERROR
lib_SetKernelArg.errcheck = _errcheck_result
def SetKernelArg(kernel, index, size, value): lib_SetKernelArg(kernel, index, size, value)

lib_EnqueueNDRangeKernel = lib.clEnqueueNDRangeKernel
lib_EnqueueNDRangeKernel.argtypes = command_queue, kernel, cl_uint, POINTER(size_t), POINTER(size_t), POINTER(size_t), cl_uint, POINTER(event), POINTER(event)
lib_EnqueueNDRangeKernel.restype  = ERROR
lib_EnqueueNDRangeKernel.errcheck = _errcheck_result
def EnqueueNDRangeKernel(queue, kernel, gsize, lsize=None, offset=None):
    nd = len(gsize)
    gsize_array = (size_t*nd)()
    for i,s in enumerate(gsize):
        gsize_array[i] = s
    if lsize is None:
        lsize_array = None
    else:
        lsize_array = (size_t*nd)()
        for i,s in enumerate(lsize):
            lsize_array[i] = s
    if offset is None:
        offset_array = None
    else:
        offset_array = (size_t*nd)()
        for i,s in enumerate(offset):
            offset_array[i] = s
    out_event = event()
    lib_EnqueueNDRangeKernel(queue, kernel, nd, offset_array, gsize_array, lsize_array, 0, None, byref(out_event))
    return out_event