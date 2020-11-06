import numpy as np
import litenn as nn
import litenn.core as nc
from litenn.core.minicl import CLDevicesBuffers

class Tensor(CLDevicesBuffers):
    """
    Represents a Tensor with given shape

    Arguments

        shape   TensorShape
                tuple of ints
                Scalar () shape is not supported.
                Use (1,) shape for scalar shape.

        init    initializer, one from nn.Init*
    """

    def backward(self, stop_grad=None, grad_for_non_trainables=False):
        """
        Perform backward computation.

            stop_grad (None)

        None or Tensor or list of Tensors where backprop should stop.

            grad_for_non_trainables (False)

        If False, the backprop will stop at those branches that have no trainable tensors (not attached to any Optimizer),
        also gradient for intermediate non trainable tensors will be freed in order to reduce memory consumption.
        """
        nn.backward(self, stop_grad=stop_grad, grad_for_non_trainables=grad_for_non_trainables)


    def has_grad(self): return self._grad_t is not None
    def get_grad(self):
        """
        Get (or creates zero) gradient of Tensor
        """
        if self._is_grad():
            raise Exception('Gradient of gradient is unsupported.')
        
        if self._grad_t is None:
            self._grad_t = TensorGrad(self.shape, init=nn.initializer.Scalar(0.0))
        return self._grad_t

    def free_grad(self):
        """
        Free gradient of Tensor
        """
        self._grad_t = None
    def zero_grad(self):
        """
        Zero gradient of Tensor
        """
        # We can do free_grad, because grad will be zeroed on first access.
        self.free_grad()

    def np_combined(self):
        """
        Returns concatenated numpy value of a Tensor from all devices along 0 axis 
        """
        np_vals = [ self.np(i) for i in range(self.np_count()) ]
        return np.concatenate(np_vals, 0)
        
    def np(self, device_id=0):
        """
        Returns numpy value of a Tensor of specified device_id
        """
        return self._get_clbuffers()[device_id].np(self.shape, dtype=np.float32)

    def np_count(self):
        """
        Returns number of devices
        """
        return len(self._get_clbuffers())

    def fill(self, value, device_id=None):
        """
        Fills whole Tensor with float value

        arguments

            value             number

            device_id(None)   id of device.
                              None - fill to all devices
        """

        value = np.float32(value)

        if device_id is not None:
            self._get_clbuffers()[device_id].fill(value)
        else:
            for device_buffer in self._get_clbuffers():
                device_buffer.fill(value)

    
    def set(self, value, device_id=None):
        """
        Set tensor value

            Parameters

            value   Tensor    copy data from Tensor. Can be with different shape, but should match size of shape.

                    Scalar number   will be treated as (1,) array

                    np.ndarray      will be converted to np.float32 dtype

            device_id   int of device_id
                        if None , set this value to all devices
        """            
        if isinstance(value, Tensor):
            if self.shape.size != value.shape.size:
                raise ValueError('Unable to set data from other tensor: shape.size is not the same.')

            for device_buffer, other_device_buffer in zip(self._get_clbuffers(), value._get_clbuffers()):
                device_buffer.set(other_device_buffer)
        else:
            if isinstance(value, np.ndarray):
                value = value.astype(np.float32)
            else:
                if isinstance(value, (int, float) ):
                    value = np.array([value], dtype=np.float32)
                elif isinstance(value, (list,tuple) ):
                    value = np.array(value, dtype=np.float32)
                else:
                    raise ValueError(f'Unknown type {value.__class__}')

            if device_id is not None:
                self._get_clbuffers()[device_id].set(value)
            else:
                for device_buffer in self._get_clbuffers():
                    device_buffer.set(value)

    def copy(self, shape=None):
        """
        Creates new tensor with the same shape and data.

            shape   override with new shape, but should match shape.size
        """
        if shape is None:
            shape = self.shape
        else:
            shape = nc.TensorShape(shape)

        if shape.size != self.shape.size:
            raise ValueError('shapes size mismatch')

        t = Tensor( shape )
        t.set(self)
        return t

    def max(self, axes=None, keepdims=False):
        """
        Reduce max operator.

            axes(None)  int
                        Iterable of ints.
                        None - all axes

            keepdims(False)     keep reduced axes
        """
        return nn.reduce_max(self, axes, keepdims=keepdims)

    def mean(self, axes=None, keepdims=False):
        """
        Reduce mean operator.

            axes(None)  int
                        Iterable of ints.
                        None - all axes

            keepdims(False)     keep reduced axes
        """
        return nn.reduce_mean(self, axes, keepdims=keepdims)

    def min(self, axes=None, keepdims=False):
        """
        Reduce min operator.

            axes(None)  int
                        Iterable of ints.
                        None - all axes

            keepdims(False)     keep reduced axes
        """
        return nn.reduce_min(self, axes, keepdims=keepdims)

    def reshape(self, target_shape):
        """
        Reshape operator.

        arguments

            target_shape   tuple of ints
        """
        return nn.reshape(self, target_shape)

    def sum(self, axes=None, keepdims=False):
        """
        Reduce sum operator.

            axes(None)  int
                        Iterable of ints.
                        None - all axes

            keepdims(False)     keep reduced axes
        """
        return nn.reduce_sum(self, axes, keepdims=keepdims)

    def transpose(self, axes_order):
        """
        Transpose operator

            axes_order      Int
                            Iterable of ints
                            None
        """
        return nn.transpose(self, axes_order)

    def T(self): 
        """
        Transpose operator with inversed axes order.
        """
        return self.transpose( tuple(range(self.shape.rank-1,-1,-1)) )
    
    def __getitem__(self, slices): return nn.slice(self, slices)

    ### INTERNAL METHODS start with _

    def __init__(self, shape, init=None):
        self.shape = shape = nc.TensorShape(shape)
        if init is None:
            init = nn.initializer.Scalar(0.0)
            
        Tensor._object_count += 1
        
        # ! internal variables (start with _) must be accesed through methods,
        # ! because Tensor can be a TensorRef that uses method overridings
        
        self._seq_id = Tensor._seq_id = Tensor._seq_id + 1        

        self._freezed = Tensor._freeze_stack != 0 # indicates that backprop
                                                  # should not go through inputs of _gradfns,
                                                  # which are marked as _is_trainable()
                                                  # use _is_freezed() to get the value

        self._trainable = False   # indicates that Tensor is used in Optimizer
                                  # use _is_trainable() to get the value

        self._grad_t = None  # Gradient Tensor of this tensor
                             # use get_grad() to get the value

        self._gradfns = None # Dict of [input_tensor] = func(O_t,dO_t), see _assign_grad
                             # use _get_gradfns() to get the value

        self._op_name = None # Name of op which produces this Tensor
        
        self._parent_module_w = None  # weakref of parent Module

        super().__init__(shape.size*4, init_func= lambda x: init.initialize_CLBuffer(x, shape)
                                       if init is not None else None)
    
    def get_name(self, top_parent=None):
        """
        returns name of the Tensor
        """
        if self._parent_module_w is None:
            name = f'#{self._get_seq_id()}'            
            if self._op_name is not None:
                name += f'({self._op_name})'
            if self._is_reference():
                name += '(ref)'
            return name
        else:
            parent_module = self._parent_module_w()    
            return f'{parent_module.get_name(top_parent)}.{parent_module._get_child_name(self)}'

        
    def __del__(self):
        Tensor._object_count -= 1
        if Tensor._object_count == 0:
            Tensor._seq_id = 0
        
    def __str__(self): return f"T {self.get_name()} {self.shape}"
    def __repr__(self):
        s = self.__str__()
        s += "\n"
        # Per device data
        buffers = self._get_clbuffers()
        for i, buffer in enumerate(buffers):
            if i == 0 or buffer is not None:
                s += str(buffer.device) + "\n"
                s += str(self.np(i)) + "\n"
        s += self.__str__()
        return s

    def __radd__(self, t):
        if isinstance(t, (int,float)):
            return nn.add_const(self, t)
        else:
            if not isinstance(t, Tensor):
                t = nn.Tensor_from_value(t)
            if t == self:
                return nn.mul_const(self, 2.0)
            else:
                return nn.add(t, self)
                
    def __add__(self, t):
        if isinstance(t, (int,float)):
            return nn.add_const(self, t)
        else:
            if not isinstance(t, Tensor):
                t = nn.Tensor_from_value(t)
            if t == self:
                return nn.mul_const(self, 2.0)
            else:
                return nn.add(self, t)
                
    def __rsub__(self, t):
        if isinstance(t, (int,float)):
            return nn.rsub_const(self, t)
        else:
            if not isinstance(t, Tensor):
                t = nn.Tensor_from_value(t)
            if t == self:
                return nn.mul_const(self, 0.0)
            else:
                return nn.sub(t, self)
            
    def __sub__(self, t):
        if isinstance(t, (int,float)):
            return nn.sub_const(self, t)
        else:
            if not isinstance(t, Tensor):
                t = nn.Tensor_from_value(t)
            if t == self:
                return nn.mul_const(self, 0.0)
            else:
                return nn.sub(self, t)
    def __rmul__(self, t):
        if isinstance(t, (int,float)):
            return nn.mul_const(self, t)
        else:
            if not isinstance(t, Tensor):
                t = nn.Tensor_from_value(t)
            if t == self:
                return nn.square(self)
            else:
                return nn.mul(t, self)
    def __mul__(self, t):
        if isinstance(t, (int,float)):
            return nn.mul_const(self, t)
        else:
            if not isinstance(t, Tensor):
                t = nn.Tensor_from_value(t)
            if t == self:
                return nn.square(self)
            else:
                return nn.mul(self, t)
    def __rtruediv__(self, t):
        if isinstance(t, (int,float)):
            return nn.rdiv_const(self, t)
        else:
            if not isinstance(t, Tensor):
                t = nn.Tensor_from_value(t)
            return nn.div(t, self)
    def __truediv__(self, t):
        if isinstance(t, (int,float)):
            return nn.div_const(self, t)
        else:
            if not isinstance(t, Tensor):
                t = nn.Tensor_from_value(t)
            return nn.div(self, t)
    def __neg__(self):
        return self * -1

    def _set_op_name(self, op_name):
        """
        set op name
        """
        self._op_name = op_name
        
    def _assign_gradfn(self, input_t, func):
        """
        Assign gradient function for 'input_t'

         input_t    Tensor

         func       callable object that receives two arguments:
                    O_t     - this Tensor
                    dO_t    - gradient of this Tensor

        func(O_t, dO_t) should add computed gradient to input_t.get_grad()

        you can pass all necessary tensors to your function using lambda,
        Tensors must NOT contain circular references.
        In this case you cannot pass this tensor inside lambda function,
        therefore you should use (O_t, dO_t) arguments passed to func by system.
        """
        grad_dict = self._gradfns
        if grad_dict is None:
            grad_dict = self._gradfns = {}

        func_ar = grad_dict.get(input_t, None)
        if func_ar is None:
            func_ar = grad_dict[input_t] = []

        func_ar.append(func)

    def _as_ref(self, shape):
        """
        Convert to Reference Tensor with new_shape.

            shape must have the same size as tensor's shape
        """
        return TensorRef(self, shape)

    def _is_freezed(self): return self._freezed
    def _is_reference(self): return False
    def _is_grad(self):      return False
    def _is_trainable(self): return self._trainable
    def _is_produced_by_op(self): return self._get_gradfns() is not None

    def _get_gradfns(self): return self._gradfns
    def _get_reference_source(self): return self
    def _get_seq_id(self): return self._seq_id
    def _get_top_reference_source(self):
        t = self
        while t._is_reference():
            t = t._get_reference_source()
        return t

    def _set_trainable(self, _trainable): self._trainable = _trainable

    _object_count = 0
    _seq_id = 0
    _freeze_stack = 0


class TensorRef(Tensor):
    """
    TensorRef used to interpret existing Tensor with different shape.
    use Tensor._as_ref() method
    """

    def __init__(self, t, shape):
        shape = nc.TensorShape(shape)
        if t.shape.size != shape.size:
            raise ValueError(f'Cannot interpet shape {t.shape} as ref shape {shape}')
        super().__init__(shape)
        self._t = t

    # Forward methods to original tensor
    def has_grad(self):  return self._t.has_grad()
    def get_grad(self):  return self._t.get_grad()._as_ref(self.shape)
    def free_grad(self): self._t.free_grad()
    def zero_grad(self): self._t.zero_grad()

    def _assign_gradfn(self, *args, **kwargs): self._t._assign_gradfn(*args, **kwargs)
    def _is_freezed(self):                     return self._t._is_freezed()
    def _is_reference(self):                   return True
    def _is_trainable(self):                   return self._t._is_trainable()
    def _get_clbuffers(self):                  return self._t._get_clbuffers()
    def _get_gradfns(self):                    return self._t._get_gradfns()
    def _get_reference_source(self):           return self._t
    def _get_seq_id(self):                     return self._t._get_seq_id()
    def _set_trainable(self, *args, **kwargs): self._t._set_trainable(*args, **kwargs)

class TensorGrad(Tensor):
    def _is_grad(self): return True

def Tensor_like(t, init=None):
    """
    Produces new Tensor with the same shape as t
    """
    return Tensor(t.shape, init=init)

def Tensor_zeros_like(t):
    """
    Produces new Tensor with the same shape as t with zeros init
    """
    return Tensor(t.shape, init=nn.initializer.Scalar(0.0))

def Tensor_ones_like(t):
    """
    Produces new Tensor with the same shape as t with ones init
    """
    return Tensor(t.shape, init=nn.initializer.Scalar(1.0))


def Tensor_from_value(value):
    """
    Produces new Tensor with the same shape as value
    and set the value to the Tensor immediately.

    arguments

     value      Tensor
                int, float, np.int32, np.float32, np.float64
                np.ndarray
    """
    if isinstance(value, list):
        value = np.array(value, np.float32)
    elif isinstance(value, (int, float, np.int32, np.float32, np.float64) ):
        value = np.array([value], np.float32)
    elif not isinstance(value, (Tensor, np.ndarray) ):
        raise ValueError(f'Unsupported value {value} ({value.__class__})')
    t = Tensor(value.shape)
    t.set(value)
    return t
    
def Tensor_sliced_from_value(value):
    """
    Produces new Tensor from np.ndarray value 
    sliced along first dimension to every current device

    arguments

     value      np.ndarray
    """
    if not isinstance(value, (Tensor, np.ndarray) ):
        raise ValueError(f'Value must be np.ndarray type')
    
    value_shape = value.shape
    if len(value_shape) < 1:
        raise ValueError(f'Shape rank must be >= 1')
    slice_axis_size = value_shape[0]
    
    devices = nn.devices.get_current()
    devices_len = len(devices)
    
    if slice_axis_size % devices_len != 0:
        raise ValueError(f'Unable to evenly slice shape {value_shape} to {devices_len} devices.')

    slice_size = slice_axis_size // devices_len
    
    tensor_shape = (slice_size,) + tuple(value_shape[1:])

    t = Tensor(tensor_shape)
    
    for i in range(devices_len):
        t.set( value[i*slice_size:(i+1)*slice_size,...], i)
        
    return t
    
