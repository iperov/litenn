import numpy as np
import litenn as nn
import litenn.core as nc
from .Optimizer import Optimizer

class Adam(Optimizer):
    """
    Adam optimizer.

    arguments

        tensor_list     list of tensors which should be optimized

        beta_1(0.9)     Adam beta_1 value >0.0 <1.0

        beta_2(0.999)   Adam beta_2 value >0.0 <1.0

        lr(0.001)       learning rate

        lr_decay(0.0)   learning rate decay

        lr_dropout(0.0) learning rate dropout.
                        [0.0 .. 1.0) probability
                        typical value is 0.7

        clipnorm(0.0)   Clip the gradient if the L2norm exceeds clipnorm.
    """

    def __init__(self, tensors_list, beta_1=0.9, beta_2=0.999, lr=0.001, lr_decay=0.0, lr_dropout=0.0, clipnorm=0.0):
        super().__init__(tensors_list, lr=lr, lr_decay=lr_decay, lr_dropout=lr_dropout, clipnorm=clipnorm,
                         saveables=['_ms', '_vs'])

        self.beta_1 = beta_1
        self.beta_2 = beta_2

        _ms, _vs = {}, {}
        for t in self.get_trainable_tensors():
            _ms[t.get_name()] = nn.Tensor_zeros_like(t)
            _vs[t.get_name()] = nn.Tensor_zeros_like(t)
        self._ms, self._vs = _ms, _vs


        self._update_ms_krn = nc.CLKernel(f"""
__kernel void impl(__global float* M, __global const float* G)
{{
    size_t gid = get_global_id(0);
    M[gid] = {beta_1}*M[gid] + (1.0 - {beta_1})*G[gid];
}}
""")
        self._update_vs_krn = nc.CLKernel(f"""
__kernel void impl(__global float* V, __global const float* G)
{{
    size_t gid = get_global_id(0);
    float g = G[gid];
    V[gid] = {beta_2}*V[gid] + (1.0 - {beta_2})*g*g;
}}
""")

        self._update_t_krn = nc.CLKernel(f"""
{self._get_lr_kernel_common_text()}


__kernel void impl (__global float* T, __global const float* M, __global const float* V
                    {self._get_lr_kernel_args_text()} )
{{
    size_t gid = get_global_id(0);

    {self._get_lr_kernel_text()}

    T[gid] += -lr*M[gid] / ( sqrt(V[gid]) + 1e-7 );
}}
""")

    def _on_step(self):
        for t in self.get_trainable_tensors():
            if not t.has_grad():
                continue
            ms_t = self._ms[t.get_name()]
            vs_t = self._vs[t.get_name()]

            self._update_ms_krn.run (ms_t, t.get_grad(), global_shape=(t.shape.size,) )
            self._update_vs_krn.run (vs_t, t.get_grad(), global_shape=(t.shape.size,) )

            self._update_t_krn.run (t, ms_t, vs_t, *self._get_lr_kernel_args(), global_shape=(t.shape.size,) )



