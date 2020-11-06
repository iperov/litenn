<table align="center" border="0">

<tr><td colspan=2 align="center">

# Developer guide.

</td></tr>
<tr><td colspan=2 align="center">

## Custom operator

</td></tr>
<tr><td colspan=2 align="left">

```python
import numpy as np
import litenn as nn
import litenn.core as nc

# define OpenCL kernels
forward_krn = nc.CLKernel("""
__kernel void impl(__global float* O, __global const float* I)
{
    int idx = get_global_id(0); //obtained from global_shape argument 
    float Iv = I[idx];
    O[idx] = Iv * (Iv >= 0);
}
""")

backward_krn = nc.CLKernel("""
__kernel void impl(__global float* dI,
                   __global const float* I,
                   __global const float* O,
                   __global const float* dO)
{
    int idx = get_global_id(0);
    dI[idx] += dO[idx] * (I[idx] >= 0);
}
""")

# define grad func (or use lambda)
# it will compute gradient for input_t argument
def relu_gradfn(O_t, dO_t):
    # You can do any intermediate computations here, 
    # using tensors as temporary GPU-memory.
    # the goal is to add computed gradient to input_t.get_grad()
    backward_krn.run(input_t.get_grad(), input_t, O_t, dO_t, 
                        global_shape=(input_t.shape.size,) )    
                        
def relu_op(input_t):
    # relu is element-wise operator
    # so we need new Tensor with the same size
    output_t = nn.Tensor(input_t.shape)    
    output_t._set_op_name('relu')
    
    # Assign gradient function
    output_t._assign_gradfn(input_t, relu_gradfn)  
    
    # Perform forward computation.
    forward_krn.run(output_t, input_t, global_shape=(input_t.shape.size,))    

    return output_t

# Testing custom op
input_t = nn.Tensor_from_value ([-3, 0, 5])
result_t = relu_op(input_t)
result_t.backward(grad_for_non_trainables=True)

print(result_t.np()) # [0. 0. 5.]
print(input_t.get_grad().np()) # [0. 1. 1.]
```

</td></tr>
<tr><td colspan=2 align="center">

## Custom element-wise operator

</td></tr>
<tr><td colspan=2 align="left">

```python
# Simpler implementation of element-wise operator

import litenn as nn
import litenn.core as nc

class my_op_kernel(nc.op.ElementWiseOpKernel):    
    def __init__(self, value): 
        self.value = value
        
    def get_forward_kernel_text(self):  
        return f"O = I*({self.value});"
        
    def get_backward_kernel_text(self): 
        return f"dI = dO*({self.value});"
        
    def get_op_name(self): 
        return f"multiply by value"
    
def my_op(input_t, value):
    return nc.op.element_wise_op(my_op_kernel, (value,), input_t)
   
# Testing custom op
input_t = nn.Tensor_from_value ([-3, 0, 5])
result_t = my_op(input_t, 2.0)
result_t.backward(grad_for_non_trainables=True)

print(result_t.np()) # [-6. 0. 10.]
print(input_t.get_grad().np()) # [2. 2. 2.]
```
---

</td></tr>

</table>
