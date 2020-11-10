import numpy as np
import litenn.core as nc
from .Initializer import Initializer
from .RandomNormal import RandomNormal
from .RandomUniform import RandomUniform


class _InitGlorot(Initializer):
    def __init__(self, type='uniform', gain=1.0, fan_in=None, fan_out=None):
        super().__init__()
        self.type = type
        self.gain = gain
        self.fan_in = fan_in
        self.fan_out = fan_out
        self.initer = None

    def has_fan_in(self): return True
    def has_fan_out(self): return True
    
    def initialize_CLBuffer(self, cl_buffer, tensor_shape):
        if self.initer is None:
            if self.fan_in is None and self.fan_out is None:
                raise ValueError(f'fan_in/fan_out should be set for {str(self)}')

            std = self.gain * np.sqrt(2.0 / (self.fan_in + self.fan_out))
            if self.type == 'uniform':
                a = np.sqrt(3.0) * std
                self.initer = RandomUniform(-a, a)
            elif self.type == 'normal':
                self.initer = RandomNormal(0.0, std)


        self.initer.initialize_CLBuffer(cl_buffer, tensor_shape)

    def __str__(self):  return f'InitGlorot type={self.type}, gain={self.gain}, fan_in={self.fan_in}, fan_out={self.fan_out}'

def GlorotUniform(gain=1.0, fan_in=None, fan_out=None):
    """
    Xavier (Glorot) uniform initializer.

    arguments

     gain(1.0)       float value

     fan_in(None)    float value
                     If None - it will be set by layer
                     where it is used (Conv2D, Dense, ...)

     fan_out(None)   float value
                     If None - it will be set by layer
                     where it is used (Conv2D, Dense, ...)


    Reference:

    "Understanding the difficulty of training deep feedforward neural networks" - Glorot, X. and Bengio, Y., using a uniform distribution.
    """

    return nc.Cacheton.get(_InitGlorot, 'uniform', gain, fan_in, fan_out)

def GlorotNormal(gain=1.0, fan_in=None, fan_out=None):
    """
    Xavier (Glorot) normal initializer.

    arguments

     gain(1.0)       float value

     fan_in(None)    float value
                     If None - it will be set by layer
                     where it is used (Conv2D, Dense, ...)

     fan_out(None)   float value
                     If None - it will be set by layer
                     where it is used (Conv2D, Dense, ...)
    """

    return nc.Cacheton.get(_InitGlorot, 'normal', gain, fan_in, fan_out)
