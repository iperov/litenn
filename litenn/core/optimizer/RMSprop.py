import numpy as np
import litenn as nn
import litenn.core as nc
from .Optimizer import Optimizer

class RMSprop(Optimizer):
    """
    RMSprop optimizer

    arguments
    
        tensor_list     list of tensors which should be optimized

        rho(0.9)        RMSprop rho value
        
        lr(0.001)       learning rate       
        
        lr_decay(0.0)   learning rate decay

        lr_dropout(0.0) learning rate dropout.
                        [0.0 .. 1.0) probability
                        typical value is 0.7
                        
        clipnorm(0.0)   Clip the gradient if the L2norm exceeds clipnorm.
    """

    def __init__(self, tensors_list, rho=0.9, lr=0.001, lr_decay=0.0, lr_dropout=0.0, clipnorm=0.0):
        super().__init__(tensors_list, lr=lr, lr_decay=lr_decay, lr_dropout=lr_dropout, clipnorm=clipnorm, 
                         saveables=['_accs'])
        
        _accs = {}
        for t in self.get_trainable_tensors():
            _accs[t.get_name()] = nn.Tensor_zeros_like(t)            
        self._accs = _accs
        self.rho = rho

        self._update_acc_krn = nc.CLKernel(f"""
__kernel void impl (__global float* A, __global const float* G)
{{
    size_t gid = get_global_id(0);
    float g = G[gid];
    A[gid] = {rho} * A[gid] + (1.0 - {rho}) * g * g;
}}
""")
        
        self._update_t_krn = nc.CLKernel(f"""
{self._get_lr_kernel_common_text()}
__kernel void impl (__global float* V, __global const float* G, __global const float* A
                    {self._get_lr_kernel_args_text()} )
{{
    size_t gid = get_global_id(0);
    
    {self._get_lr_kernel_text()}
    
    V[gid] += -lr * G[gid] / ( sqrt(A[gid]) + 1e-7 );
}}
""")
    
    def _on_step(self):        
        for t in self.get_trainable_tensors():
            if not t.has_grad():
                continue
            acc_t = self._accs[t.get_name()]
                
            self._update_acc_krn.run (acc_t, t.get_grad(),  global_shape=(t.shape.size,) )
            self._update_t_krn.run (t, t.get_grad(), acc_t, *self._get_lr_kernel_args(), global_shape=(t.shape.size,) )
