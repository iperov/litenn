import numpy as np
import litenn as nn
import litenn.core as nc


class DepthwiseConv2D(nn.Module):
    """
    DepthwiseConv2D module.

        Parameters

        in_ch       int             input channels
        kernel_size int             kernel size
        stride      int             default(1)
        dilation    int             default(1)
        padding     'valid'         no padding
                    'same'          output size will be the same or divided by stride
                    int             padding value for all sides
                    tuple of 4 ints paddings for left,top,right,bottom sides

        use_bias    bool            use bias or not

        kernel_initializer(None)   initializer for kernel, 
                                      default - from hint or nn.initializer.GlorotUniform()
                                      
        bias_initializer(None)     initializer for bias,   default - nn.initializer.Scalar(0.0)
    """
    def __init__(self, in_ch, kernel_size, stride=1, dilation=1, padding='same', use_bias=True, depthwise_initializer=None, pointwise_initializer=None, bias_initializer=None ):

        self.in_ch = in_ch
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.use_bias = use_bias

        if depthwise_initializer is None:
            depthwise_initializer = nc.Cacheton.get_var('DepthwiseConv2D_default_depthwise_initializer')
        if depthwise_initializer is None:
            depthwise_initializer = nn.initializer.GlorotUniform()
        if depthwise_initializer.has_fan_in():
            if depthwise_initializer.fan_in is None: depthwise_initializer.fan_in  = in_ch  * kernel_size * kernel_size
        if depthwise_initializer.has_fan_out():
            if depthwise_initializer.fan_out is None: depthwise_initializer.fan_out = 0

        self.depthwise_initializer = depthwise_initializer
        self.kernel = nn.Tensor( (in_ch, kernel_size, kernel_size), init=depthwise_initializer )
        
        self.bias = None
        if self.use_bias:
            if bias_initializer is None:
                bias_initializer = nn.initializer.Scalar(0.0)
            self.bias_initializer = bias_initializer
            self.bias = nn.Tensor( (self.in_ch,), init=bias_initializer)

        super().__init__(saveables=['kernel','bias'])

    def forward(self, x, **kwargs):
        x = nn.depthwise_conv2D(x, self.kernel, self.stride, self.dilation, padding=self.padding)
        
        if self.use_bias:
            x = x + self.bias.reshape( (1,-1,1,1) )    
        return x

    def __str__(self): return f"{self.__class__.__name__} : in_ch:{self.in_ch} "
    def __repr__(self): return self.__str__()

def hint_DepthwiseConv2D_default_depthwise_initializer(depthwise_initializer):
    """
    set default 'depthwise_initializer' for DepthwiseConv2D.

    it will be reseted after call nn.cleanup()
    """
    if not isinstance(depthwise_initializer, nc.initializer.Initializer):
        raise ValueError(f'depthwise_initializer should be class of nn.initializer.*')

    nc.Cacheton.set_var('DepthwiseConv2D_default_depthwise_initializer', depthwise_initializer)

def DepthwiseConv2D_test():
    module = DepthwiseConv2D(4,3)        
    x = nn.Tensor( (2,4,8,8) )
    y = module(x)
    y.backward(grad_for_non_trainables=True)
    if not x.has_grad():
        raise Exception('x has no grad')