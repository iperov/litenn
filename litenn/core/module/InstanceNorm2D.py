import numpy as np
import litenn as nn
import litenn.core as nc


class InstanceNorm2D(nn.Module):
    """
    Instance Normalization 2D module.

        Parameters

        in_ch       int             input channels

    """
    def __init__(self, in_ch):
        self.in_ch = in_ch

        self.gamma = nn.Tensor( (in_ch,), init=nn.initializer.Scalar(1.0) )
        self.beta  = nn.Tensor( (in_ch,), init=nn.initializer.Scalar(0.0) )

        super().__init__(saveables=['gamma','beta'])

    def forward(self, x, **kwargs):

        mean, var = nn.moments(x, axes=(2,3), keepdims=True)

        x = x-mean / (nn.sqrt(var) + 1e-5)

        x = x * self.gamma.reshape( (1,-1,1,1) ) \
              + self.beta.reshape( (1,-1,1,1) )
        return x

    def __str__(self): return f"{self.__class__.__name__} : in_ch:{self.in_ch}"
    def __repr__(self): return self.__str__()

def InstanceNorm2D_test():
    module = InstanceNorm2D(4)
    x = nn.Tensor( (2,4,8,8) )
    y = module(x)
    y.backward(grad_for_non_trainables=True)