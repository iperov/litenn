import numpy as np
import litenn as nn
import litenn.core as nc


class AdaptiveInstanceNorm2D(nn.Module):
    """
    Adaptive Instance Normalization 2D module.

    arguments

        in_ch       int         input channels

        in_mlp_ch   int         input MLP channels
                                (BATCH,in_mlp_ch)

        gamma_initializer(None)
                    initializer for gamma, default - from hint or nn.initializer.HeNormal()

    forward(x) inputs

        x = [ NCHW , NC ]

        where   NCHW  - input 2D tensor
                NC    - input MLP tensor, must match in_mlp_ch

    reference

    Arbitrary Style Transfer in Real-time with Adaptive Instance Normalization
    https://arxiv.org/abs/1703.06868
    """
    def __init__(self, in_ch, in_mlp_ch, gamma_initializer=None):
        self.in_ch = in_ch
        self.in_mlp_ch = in_mlp_ch

        if gamma_initializer is None:
            gamma_initializer = nc.Cacheton.get_var('AdaptiveInstanceNorm2D_default_gamma_initializer')
        if gamma_initializer is None:
            gamma_initializer = nn.initializer.HeNormal()
        if gamma_initializer.has_fan_in():
            if gamma_initializer.fan_in is None: gamma_initializer.fan_in = in_ch
        if gamma_initializer.has_fan_out():
            if gamma_initializer.fan_out is None: gamma_initializer.fan_out = in_ch

        self.gamma1 = nn.Tensor( (in_mlp_ch,in_ch,), init=gamma_initializer )
        self.beta1  = nn.Tensor( (in_ch,), init=nn.initializer.Scalar(0.0) )
        self.gamma2 = nn.Tensor( (in_mlp_ch,in_ch,), init=gamma_initializer )
        self.beta2  = nn.Tensor( (in_ch,), init=nn.initializer.Scalar(0.0) )

        super().__init__(saveables=['gamma1','beta1','gamma2','beta2'])

    def forward(self, inputs, **kwargs):
        x, mlp = inputs

        gamma = nn.matmul(mlp, self.gamma1)
        gamma = gamma + self.beta1

        beta = nn.matmul(mlp, self.gamma2)
        beta = beta + self.beta2

        mean, var = nn.moments(x, axes=(2,3), keepdims=True)

        x = x-mean / (nn.sqrt(var) + 1e-5)

        x = x * gamma.reshape( (-1,self.in_ch,1,1) ) \
              + beta.reshape( (-1,self.in_ch,1,1) )
        return x

    def __str__(self): return f"{self.__class__.__name__} : in_ch:{self.in_ch} in_mlp_ch:{self.in_mlp_ch}"
    def __repr__(self): return self.__str__()

def hint_AdaptiveInstanceNorm2D_default_gamma_initializer(gamma_initializer):
    """
    set default 'gamma_initializer' for AdaptiveInstanceNorm2D.

    it will be reseted after call nn.cleanup()
    """
    if not isinstance(gamma_initializer, nc.initializer.Initializer):
        raise ValueError(f'gamma_initializer should be class of nn.initializer.*')

    nc.Cacheton.set_var('AdaptiveInstanceNorm2D_default_gamma_initializer', gamma_initializer)


def AdaptiveInstanceNorm2D_test():
    mlp = nn.Tensor( (2,8) )

    module = AdaptiveInstanceNorm2D(4, 8)
    x = nn.Tensor( (2,4,8,8) )
    y = module([x, mlp])
    y.backward(grad_for_non_trainables=True)

    if not mlp.has_grad():
        raise Exception('mlp has no grad')
    if not x.has_grad():
        raise Exception('x has no grad')