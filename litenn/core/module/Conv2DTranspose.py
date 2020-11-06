import numpy as np
import litenn as nn
import litenn.core as nc

class Conv2DTranspose(nn.Module):
    """
    Conv2DTranspose module.

    arguments
        
        in_ch       int             input channels
        out_ch      int             output channels
        kernel_size int             kernel size
        stride      int             default(2)
        dilation    int             default(1)
        padding     'valid'         no padding
                    'same'          output size will be the same or divided by stride

        use_bias    bool            use bias or not
        
        kernel_initializer(None)   initializer for kernel, default - from hint or nn.initializer.GlorotUniform()
        bias_initializer(None)     initializer for bias,   default - nn.initializer.Scalar(0.0)
    """
    def __init__(self, in_ch, out_ch, kernel_size, stride=2, dilation=1, padding='same', use_bias=True, kernel_initializer=None, bias_initializer=None ):

        self.in_ch = in_ch
        self.out_ch = out_ch
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.use_bias = use_bias

        if kernel_initializer is None:
            kernel_initializer = nc.Cacheton.get_var('Conv2DTranspose_default_kernel_initializer')
        if kernel_initializer is None:
            kernel_initializer = nn.initializer.GlorotUniform()        
        if kernel_initializer.has_fan_in_out():
            if kernel_initializer.fan_in is None: kernel_initializer.fan_in  = in_ch  * kernel_size * kernel_size
            if kernel_initializer.fan_out is None: kernel_initializer.fan_out = out_ch * kernel_size * kernel_size
                
        self.kernel_initializer = kernel_initializer
        self.kernel = nn.Tensor( (out_ch, in_ch, kernel_size, kernel_size), init=kernel_initializer )
        
        self.bias = None
        if self.use_bias:
            if bias_initializer is None:
                bias_initializer = nn.initializer.Scalar(0.0)
            self.bias_initializer = bias_initializer    
            self.bias = nn.Tensor( (self.out_ch,), init=bias_initializer)
            
        super().__init__(saveables=['kernel','bias'])

    def forward(self, x, **kwargs):
        x = nn.conv2DTranspose(x, self.kernel, self.stride, self.dilation, padding=self.padding)
        if self.use_bias:
            x = x + self.bias.reshape( (1,-1,1,1) )
        return x

    def __str__(self): return f"{self.__class__.__name__} : in_ch:{self.in_ch} out_ch:{self.out_ch} "
    def __repr__(self): return self.__str__()
    
def hint_Conv2DTranspose_default_kernel_initializer(kernel_initializer):
    """
    set default 'kernel_initializer' for Conv2DTranspose.
    
    it will be reseted after call nn.cleanup()
    """
    if not isinstance(kernel_initializer, nc.initializer.Initializer):
        raise ValueError(f'kernel_initializer should be class of nn.initializer.*')
    
    nc.Cacheton.set_var('Conv2DTranspose_default_kernel_initializer', kernel_initializer)
    
def Conv2DTranspose_test():
    module = Conv2DTranspose(4,8,3)
    
    x = nn.Tensor( (2,4,8,8) )
    y = module(x)
    y.backward(grad_for_non_trainables=True)