import numpy as np
import litenn as nn
import litenn.core as nc


class BatchNorm2D(nn.Module):
    """
    Batch Normalization 2D module.

    arguments

        in_ch       int             input channels
    
    Don't forget to call most parent Module.set_training(bool)
    
    References
    
    [Batch Normalization: Accelerating Deep Network Training by
    Reducing Internal Covariate Shift](https://arxiv.org/abs/1502.03167)
    """
    def __init__(self, in_ch, momentum=0.99):
        self.in_ch = in_ch
        
        if momentum < 0 or momentum > 1.0:
            raise ValueError(f'momentum {momentum} must be in range [0 .. 1.0]')
        self.momentum = momentum

        self.gamma = nn.Tensor( (in_ch,), init=nn.initializer.Scalar(1.0) )
        self.beta  = nn.Tensor( (in_ch,), init=nn.initializer.Scalar(0.0) )
        
        self.running_mean = nn.Tensor( (in_ch,), init=nn.initializer.Scalar(0.0) )
        self.running_var  = nn.Tensor( (in_ch,), init=nn.initializer.Scalar(1.0) )

        super().__init__(saveables=['gamma','beta','running_mean','running_var'],
                         trainables=['gamma','beta'] )

    def forward(self, x, **kwargs):
        
        if self.is_training():
            mean, var = nn.moments(x, axes=(0,2,3), keepdims=True)
            
            BatchNorm2D.upd_krn.run(self.running_mean, self.running_var, mean, var, np.float32(self.momentum), global_shape=(self.in_ch,))
        else:
            mean = self.running_mean.reshape( (1,-1,1,1) )     
            var = self.running_var.reshape( (1,-1,1,1) )     

        x = (x-mean) / (nn.sqrt(var) + 1e-5)
        
        x = x * self.gamma.reshape( (1,-1,1,1) ) \
              + self.beta.reshape( (1,-1,1,1) )    
        return x

    def __str__(self): return f"{self.__class__.__name__} : in_ch:{self.in_ch}"
    def __repr__(self): return self.__str__()


    upd_krn = nc.CLKernel(kernel_text="""
__kernel void asd( __global float* RM, __global float* RV, __global const float* M, __global const float* V, float momentum)
{{
    size_t gid = get_global_id(0);
    
    float rm = RM[gid];
    float m = M[gid];
    float rv = RV[gid];
    float v = V[gid];
    
    RM[gid] = rm*momentum + m*(1.0-momentum);
    RV[gid] = rv*momentum + v*(1.0-momentum);
}}
""")


def BatchNorm2D_test():
    module = BatchNorm2D(4)        
    module.set_training(True)
    
    x = nn.Tensor( (2,4,8,8) )
    y = module(x)
    y.backward(grad_for_non_trainables=True)
    
    if not x.has_grad():
        raise Exception('x has no grad')
    
    module.set_training(False)        
    x = nn.Tensor( (2,4,8,8) )
    y = module(x)
    y.backward(grad_for_non_trainables=True)
    
    