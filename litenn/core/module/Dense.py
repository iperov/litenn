import numpy as np
import litenn as nn
import litenn.core as nc

class Dense(nn.Module):
    """
    Dense layer.

        Parameters
        
        in_ch           int             input channels
        out_ch          int             output channels
        use_bias        bool            use bias or not
        
        weight_initializer     initializer for weight, default - from hint or nn.initializer.GlorotUniform()
        bias_initializer       initializer for bias,   default - nn.initializer.Scalar(0.0)      
    """
    def __init__(self, in_ch, out_ch, use_bias=True, weight_initializer=None, bias_initializer=None ):

        self.in_ch = in_ch
        self.out_ch = out_ch
        self.use_bias = use_bias
        
        if weight_initializer is None:
            weight_initializer = nc.Cacheton.get_var('Dense_default_weight_initializer')
        if weight_initializer is None:
            weight_initializer = nn.initializer.GlorotUniform()            
        if weight_initializer.has_fan_in_out():
            if weight_initializer.fan_in is None: weight_initializer.fan_in = in_ch
            if weight_initializer.fan_out is None: weight_initializer.fan_out = out_ch

        self.weight = nn.Tensor( (in_ch, out_ch), init=weight_initializer )
        
        self.bias = None
        if self.use_bias:
            if bias_initializer is None:
                bias_initializer = nn.initializer.Scalar(0.0)
            self.bias_initializer = bias_initializer    
            self.bias = nn.Tensor( (self.out_ch,), init=bias_initializer)
            
        super().__init__(saveables=['weight','bias'])

    def forward(self, x, **kwargs):
        x = nn.matmul(x, self.weight)
        if self.use_bias:
            x = x + self.bias
        return x

    def __str__(self): return f"{self.__class__.__name__} : in_ch:{self.in_ch} out_ch:{self.out_ch} "
    def __repr__(self): return self.__str__()
    
def hint_Dense_default_weight_initializer(weight_initializer):
    """
    set default 'weight_initializer' for Dense.
    
    it will be reseted after call nn.cleanup()
    """
    if not isinstance(weight_initializer, nc.initializer.Initializer):
        raise ValueError(f'weight_initializer should be class of nn.initializer.*')
    
    nc.Cacheton.set_var('Dense_default_weight_initializer', weight_initializer)
    
def Dense_test():
    module = Dense(4,8)        
    x = nn.Tensor( (2,4) )
    y = module(x)
    y.backward(grad_for_non_trainables=True)