import numpy as np
import litenn.core as nc
from .Initializer import Initializer
from .RandomNormal import RandomNormal
from .RandomUniform import RandomUniform


class _InitHe(Initializer):
    def __init__(self, type='uniform', gain=2.0, fan_in=None):
        super().__init__()
        self.type = type
        self.gain = gain
        self.fan_in = fan_in
        self.initer = None

    def has_fan_in(self): return True
    
    def initialize_CLBuffer(self, cl_buffer, tensor_shape):
        if self.initer is None:
            if self.fan_in is None:
                raise ValueError(f'fan_in should be set for {str(self)}')

            std = self.gain * np.sqrt(2.0 / self.fan_in)
            if self.type == 'uniform':
                a = np.sqrt(3.0) * std
                self.initer = RandomUniform(-a, a)
            elif self.type == 'normal':
                self.initer = RandomNormal(0.0, std)


        self.initer.initialize_CLBuffer(cl_buffer, tensor_shape)

    def __str__(self):  return f'InitHe type={self.type}, gain={self.gain}, fan_in={self.fan_in}'

def HeUniform(gain=2.0, fan_in=None):
    """
    He uniform initializer.

    arguments

     gain(1.0)       float value

     fan_in(None)    float value
                     If None - it will be set by layer
                     where it is used (Conv2D, Dense, ...)

    """

    return nc.Cacheton.get(_InitHe, 'uniform', gain, fan_in)

def HeNormal(gain=1.0, fan_in=None):
    """
    He normal initializer.

    arguments

     gain(1.0)       float value

     fan_in(None)    float value
                     If None - it will be set by layer
                     where it is used (Conv2D, Dense, ...)
    """

    return nc.Cacheton.get(_InitHe, 'normal', gain, fan_in)
