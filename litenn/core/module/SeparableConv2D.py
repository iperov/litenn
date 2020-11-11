import numpy as np
import litenn as nn
import litenn.core as nc


class SeparableConv2D(nn.Module):
    """
    SeparableConv2D module.

        Parameters

        in_ch       int             input channels
        out_ch      int             output channels
        kernel_size int             kernel size
        stride      int             default(1)
        dilation    int             default(1)
        padding     'valid'         no padding
                    'same'          output size will be the same or divided by stride
                    int             padding value for all sides
                    tuple of 4 ints paddings for left,top,right,bottom sides

        use_bias    bool            use bias or not

        depthwise_initializer(None)   initializer for depthwise_kernel, 
                                      default - from hint or nn.initializer.GlorotUniform()
                                      
        pointwise_initializer(None)   initializer for pointwise_kernel, 
                                      default - from hint or nn.initializer.GlorotUniform()
                                      
        bias_initializer(None)     initializer for bias,   default - nn.initializer.Scalar(0.0)
    """
    def __init__(self, in_ch, out_ch, kernel_size, stride=1, dilation=1, padding='same', use_bias=True, depthwise_initializer=None, pointwise_initializer=None, bias_initializer=None ):

        self.in_ch = in_ch
        self.out_ch = out_ch
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.use_bias = use_bias

        if depthwise_initializer is None:
            depthwise_initializer = nc.Cacheton.get_var('SeparableConv2D_default_depthwise_initializer')
        if depthwise_initializer is None:
            depthwise_initializer = nn.initializer.GlorotUniform()
        if depthwise_initializer.has_fan_in():
            if depthwise_initializer.fan_in is None: depthwise_initializer.fan_in  = in_ch  * kernel_size * kernel_size
        if depthwise_initializer.has_fan_out():
            if depthwise_initializer.fan_out is None: depthwise_initializer.fan_out = 0

        self.depthwise_initializer = depthwise_initializer
        self.depthwise_kernel = nn.Tensor( (in_ch, kernel_size, kernel_size), init=depthwise_initializer )
        
        
        if pointwise_initializer is None:
            pointwise_initializer = nc.Cacheton.get_var('SeparableConv2D_default_pointwise_initializer')
        if pointwise_initializer is None:
            pointwise_initializer = nn.initializer.GlorotUniform()
        if pointwise_initializer.has_fan_in():
            if pointwise_initializer.fan_in is None: pointwise_initializer.fan_in  = in_ch
        if pointwise_initializer.has_fan_out():
            if pointwise_initializer.fan_out is None: pointwise_initializer.fan_out = out_ch

        self.pointwise_initializer = pointwise_initializer
        self.pointwise_kernel = nn.Tensor( (out_ch, in_ch, 1, 1), init=pointwise_initializer )

        self.bias = None
        if self.use_bias:
            if bias_initializer is None:
                bias_initializer = nn.initializer.Scalar(0.0)
            self.bias_initializer = bias_initializer
            self.bias = nn.Tensor( (self.out_ch,), init=bias_initializer)

        super().__init__(saveables=['depthwise_kernel','pointwise_kernel','bias'])

    def forward(self, x, **kwargs):
        x = nn.depthwise_conv2D(x, self.depthwise_kernel, self.stride, self.dilation, padding=self.padding)
        x = nn.conv2D(x, self.pointwise_kernel, 1, 1, padding='valid')
        
        if self.use_bias:
            x = x + self.bias.reshape( (1,-1,1,1) )    
        return x

    def __str__(self): return f"{self.__class__.__name__} : in_ch:{self.in_ch} out_ch:{self.out_ch} "
    def __repr__(self): return self.__str__()

def hint_SeparableConv2D_default_depthwise_initializer(depthwise_initializer):
    """
    set default 'depthwise_initializer' for SeparableConv2D.

    it will be reseted after call nn.cleanup()
    """
    if not isinstance(depthwise_initializer, nc.initializer.Initializer):
        raise ValueError(f'depthwise_initializer should be class of nn.initializer.*')

    nc.Cacheton.set_var('SeparableConv2D_default_depthwise_initializer', depthwise_initializer)
    
def hint_SeparableConv2D_default_pointwise_initializer(pointwise_initializer):
    """
    set default 'pointwise_initializer' for SeparableConv2D.

    it will be reseted after call nn.cleanup()
    """
    if not isinstance(pointwise_initializer, nc.initializer.Initializer):
        raise ValueError(f'pointwise_initializer should be class of nn.initializer.*')

    nc.Cacheton.set_var('SeparableConv2D_default_pointwise_initializer', pointwise_initializer)


def SeparableConv2D_test():
    module = SeparableConv2D(4,8,3)        
    x = nn.Tensor( (2,4,8,8) )
    y = module(x)
    y.backward(grad_for_non_trainables=True)
    if not x.has_grad():
        raise Exception('x has no grad')