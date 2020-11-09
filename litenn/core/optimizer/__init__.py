import litenn as nn

from .Optimizer import Optimizer
from .AdaBelief import AdaBelief
from .Adam import Adam
from .RMSprop import RMSprop
from .SGD import SGD

class freeze:
    """
    tensors marked as trainable(attached to Optimizer)
    that are used to create any operations inside block
        with optimizer.freeze():
            ...
    will not be trained
    """
    def __enter__(self):
        nn.Tensor._freeze_stack += 1

    def __exit__(self, a,b,c):
        nn.Tensor._freeze_stack -= 1
        if nn.Tensor._freeze_stack < 0:
            raise ValueError('Wrong _freeze_stack')
