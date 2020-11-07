import traceback

import numpy as np

import litenn as nn
import litenn.core as nc
from litenn.core import CLKernelHelper as ph


class DualWiseOpKernel:
    """
    Base class for kernels to use in dual_wise_op()
    """
    def get_forward_kernel_text(self):
        """
        return kernel C code block for forward operation
        This block will be inserted to the complete OpenCL kernel code.

        available variables:
            A - input A for forward
            B - input B for forward
            O - store result of forward operation

        You can declare and use any intermediate variables.

        example code block for mul operation:

            return "O = A * B;"
        """
        raise NotImplementedError()

    def get_backward_A_kernel_text(self):
        """
        return kernel C code for backward operation
        This block will be inserted to the complete OpenCL kernel code.

        available variables:
            A - input A for forward
            B - input B for forward
            O - result of forward
            dO - gradient of O
            
            dA - store result of backward for input A

        example code block for backward mul operation:

            return "dA = dO * B;"
        """
        raise NotImplementedError()
        
    def get_backward_B_kernel_text(self):
        """
        return kernel C code for backward operation
        This block will be inserted to the complete OpenCL kernel code.

        available variables:
            A - input A for forward
            B - input B for forward
            O - result of forward
            dO - gradient of O
            
            dB - store result of backward for input B

        example code block for backward mul operation:

            return "dB = dO * A;"
        """
        raise NotImplementedError()
    
    def get_op_name(self):
        raise NotImplementedError()

    def __str__(self): return f'DualWiseOpKernel ({self.get_op_name()})'
    def __repr__(self): return self.__str__()


def dual_wise_op(DualWiseOpKernel_cls, DualWiseOpKernel_args, a_t, b_t, output_t=None, is_add_to_output=False):
    """
    operator for DualWiseOpKernel ops with two inputs
    
    arguments

        DualWiseOpKernel_cls     class derived from DualWiseOpKernel

        DualWiseOpKernel_args    args to construct DualWiseOpKernel_cls

        output_t            compute result to this Tensor.
                            Tensor may be with different shape, but should match total size.
                            gradfn will not be set.

        is_add_to_output    add result to output_t if output_t is set.
    """
    is_add_to_output = False if output_t is None else is_add_to_output


    op = nc.Cacheton.get(_Input2Op, DualWiseOpKernel_cls, DualWiseOpKernel_args, a_t.shape, b_t.shape, is_add_to_output)

    if output_t is None:
        output_t = nn.Tensor ( op.info.output_shape )
        output_t._set_op_name (f'{op.kernel.get_op_name()}')
        output_t._assign_gradfn (a_t, lambda O_t, dO_t: dual_wise_op_A_gradfn(op, a_t, b_t, O_t, dO_t) )
        output_t._assign_gradfn (b_t, lambda O_t, dO_t: dual_wise_op_B_gradfn(op, a_t, b_t, O_t, dO_t) )
    elif output_t.shape.size != op.info.output_shape.size:
        raise ValueError(f'output_t must have size { op.info.output_shape.size }')

    op.forward_krn.run(output_t, a_t, b_t)

    return output_t

def dual_wise_op_A_gradfn(op, a_t, b_t, O_t, dO_t):
    dA_t = a_t.get_grad()
    ared = op.info.a_shape_reduction_axes
    if ared.rank != 0:
        # a_t should be reducted from O_t, thus compute gradient to temporary tensor
        # and then reduce_sum it to a_t.grad
        dA_t = nn.Tensor_zeros_like(dO_t)
    op.backward_A_krn.run(dA_t, a_t, b_t, O_t, dO_t)
    if ared.rank != 0:
        nc.op.reduce_sum (dA_t, ared, output_t=a_t.get_grad(), is_add_to_output=True)

def dual_wise_op_B_gradfn(op, a_t, b_t, O_t, dO_t):
    dB_t = b_t.get_grad()
    bred = op.info.b_shape_reduction_axes
    if bred.rank != 0:
        # b_t should be reducted from O_t, thus compute gradient to temporary tensor
        # and then reduce_sum it to b_t.grad
        dB_t = nn.Tensor_zeros_like(dO_t)
    op.backward_B_krn.run(dB_t, a_t, b_t, O_t, dO_t)
    if bred.rank != 0:
        nc.op.reduce_sum (dB_t, bred, output_t=b_t.get_grad(), is_add_to_output=True)

class _Input2Op:
    def __init__(self, DualWiseOpKernel_cls, DualWiseOpKernel_args, a_shape : nc.TensorShape, b_shape : nc.TensorShape, is_add_to_output):
        self.kernel = DualWiseOpKernel_cls(*DualWiseOpKernel_args)
        self.info = info = nc.info.InfoBroadcast(a_shape, b_shape)

        # Implement kernel. Process of both broadcasted shapes using index modulus accessor.
        self.forward_krn = nc.CLKernel(global_shape=(info.output_shape.size,), kernel_text=f"""
{ph.define_axes_accessor('A', info.a_br_shape )}
{ph.define_axes_accessor('B', info.b_br_shape )}
{ph.define_axes_accessor('O', info.output_shape )}
__kernel void impl(__global float* O_t, __global const float* A_t, __global const float* B_t)
{{
size_t gid = get_global_id(0);
{ph.axes_idxs_from_var('o', info.output_shape.rank, 'gid')}
float A = A_t[A_idx_mod({ph.axes_seq_enum('o', info.output_shape.rank)})];
float B = B_t[B_idx_mod({ph.axes_seq_enum('o', info.output_shape.rank)})];
float O = 0.0;
{self.kernel.get_forward_kernel_text()}
O_t[get_global_id(0)] {'+=' if is_add_to_output else '='} O;
}}
""")

        self.backward_A_krn = nc.CLKernel(global_shape=(info.output_shape.size,), kernel_text=f"""
{ph.define_axes_accessor('A', info.a_br_shape)}
{ph.define_axes_accessor('B', info.b_br_shape)}
{ph.define_axes_accessor('O', info.output_shape)}
__kernel void impl(__global float* dA_t,
                   __global const float* A_t, __global const float* B_t,
                   __global const float* O_t, __global const float* dO_t)
{{
size_t gid = get_global_id(0);
{ph.axes_idxs_from_var('o', info.output_shape.rank, 'gid')}
float A = A_t[A_idx_mod({ph.axes_seq_enum('o', info.output_shape.rank)})];
float B = B_t[B_idx_mod({ph.axes_seq_enum('o', info.output_shape.rank)})];
float O = O_t[gid];
float dO = dO_t[gid];
float dA = 0.0;
{self.kernel.get_backward_A_kernel_text()}
dA_t[gid] += dA;
}}
""")
        self.backward_B_krn = nc.CLKernel(global_shape=(info.output_shape.size,), kernel_text=f"""
{ph.define_axes_accessor('A', info.a_br_shape)}
{ph.define_axes_accessor('B', info.b_br_shape)}
{ph.define_axes_accessor('O', info.output_shape)}
__kernel void impl(__global float* dB_t,
                   __global const float* A_t, __global const float* B_t,
                   __global const float* O_t, __global const float* dO_t)
{{
size_t gid = get_global_id(0);
{ph.axes_idxs_from_var('o', info.output_shape.rank, 'gid')}
float A = A_t[A_idx_mod({ph.axes_seq_enum('o', info.output_shape.rank)})];
float B = B_t[B_idx_mod({ph.axes_seq_enum('o', info.output_shape.rank)})];
float O = O_t[gid];
float dO = dO_t[gid];
float dB = 0.0;
{self.kernel.get_backward_B_kernel_text()}
dB_t[gid] += dB;
}}
""")


class add_kernel(DualWiseOpKernel):
    def get_forward_kernel_text(self):  return f"O = A + B;"
    def get_backward_A_kernel_text(self): return f"dA = dO;"
    def get_backward_B_kernel_text(self): return f"dB = dO;"
    def get_op_name(self): return f"add"
def add_op(a_t, b_t, output_t=None, is_add_to_output=False): return dual_wise_op(add_kernel, (), a_t, b_t, output_t=output_t, is_add_to_output=is_add_to_output)
def add(a_t, b_t):
    """
    add operator a+b
    """
    return add_op(a_t, b_t)
    
class binary_crossentropy_kernel(DualWiseOpKernel):
    def get_forward_kernel_text(self):  return f"O = -B * log(A) - (1-B) * log(1-A);"
    def get_backward_A_kernel_text(self): return f"dA = dO * (A-B) / (A*(1-A));"
    def get_backward_B_kernel_text(self): return f"dB = dO;"
    def get_op_name(self): return f"binary_crossentropy"
def binary_crossentropy_op(input_t, target_t, output_t=None, is_add_to_output=False): return dual_wise_op(binary_crossentropy_kernel, (), input_t, target_t, output_t=output_t, is_add_to_output=is_add_to_output)
def binary_crossentropy(input_t, target_t):
    """
    Binary cross entropy operation.
    
    arguments
        
        input_t     Tensor
                    must be the result 
                    of sigmoid operation.
                    
        target_t    Tensor
                    target probabilities
    
    """
    return binary_crossentropy_op(input_t, target_t)

def categorical_crossentropy(input_t, target_t):
    """
    Categorical cross entropy operation.
    
    arguments
        
        input_t     Tensor of rank 2 (BATCH,C)
                    must be the result 
                    of softmax operation.
                    
        target_t    Tensor of rank 1 (C,)
                    or rank 2 (BATCH,C)
                    of target probabilities
    """
    if input_t.shape.rank != 2:
        raise ValueError(f'input_t.shape.rank != 2')
    N,C = input_t.shape
    
    if not target_t.shape.rank in (1,2):
        raise ValueError(f'target_t.shape.rank != 1 or 2')
    
    input_t = nn.clip(input_t, 1e-7, 1 - 1e-7)
    return nn.reduce_sum( -target_t * nn.log(input_t), axes=-1)



class sub_kernel(DualWiseOpKernel):
    def get_forward_kernel_text(self):  return f"O = A - B;"
    def get_backward_A_kernel_text(self): return f"dA = dO;"
    def get_backward_B_kernel_text(self): return f"dB = -dO;"
    def get_op_name(self): return f"sub"
def sub_op(a_t, b_t, output_t=None, is_add_to_output=False): return dual_wise_op(sub_kernel, (), a_t, b_t, output_t=output_t, is_add_to_output=is_add_to_output)
def sub(a_t, b_t):
    """
    sub operator a-b
    """
    return sub_op(a_t, b_t)
     
class max_kernel(DualWiseOpKernel):
    def get_forward_kernel_text(self):  return f"O = fmax(A,B);"
    def get_backward_A_kernel_text(self): return f"dA = dO * (A >= B);"
    def get_backward_B_kernel_text(self): return f"dB = dO * (A < B);"
    def get_op_name(self): return f"max"
def max_op(a_t, b_t, output_t=None, is_add_to_output=False): return dual_wise_op(max_kernel, (), a_t, b_t, output_t=output_t, is_add_to_output=is_add_to_output)
def max(a_t, b_t):
    """
    max operator
    """
    return max_op(a_t, b_t)

class min_kernel(DualWiseOpKernel):
    def get_forward_kernel_text(self):  return f"O = fmin(A,B);"
    def get_backward_A_kernel_text(self): return f"dA = dO * (A <= B);"
    def get_backward_B_kernel_text(self): return f"dB = dO * (A > B);"
    def get_op_name(self): return f"min"
def min_op(a_t, b_t, output_t=None, is_add_to_output=False): return dual_wise_op(min_kernel, (), a_t, b_t, output_t=output_t, is_add_to_output=is_add_to_output)
def min(a_t, b_t):
    """
    min operator 
    """
    return min_op(a_t, b_t)
     
class mul_kernel(DualWiseOpKernel):
    def get_forward_kernel_text(self):  return f"O = A * B;"
    def get_backward_A_kernel_text(self): return f"dA = dO*B;"
    def get_backward_B_kernel_text(self): return f"dB = dO*A;"
    def get_op_name(self): return f"mul"
def mul_op(a_t, b_t, output_t=None, is_add_to_output=False): return dual_wise_op(mul_kernel, (), a_t, b_t, output_t=output_t, is_add_to_output=is_add_to_output)
def mul(a_t, b_t):
    """
    mul operator a*b
    """
    return mul_op(a_t, b_t)
    
class div_kernel(DualWiseOpKernel):
    def get_forward_kernel_text(self):  return f"O = A / B;"
    def get_backward_A_kernel_text(self): return f"dA = dO / B;"
    def get_backward_B_kernel_text(self): return f"dB = dO * ( -(A / (B*B)));"
    def get_op_name(self): return f"div"
def div_op(a_t, b_t, output_t=None, is_add_to_output=False): return dual_wise_op(div_kernel, (), a_t, b_t, output_t=output_t, is_add_to_output=is_add_to_output)
def div(a_t, b_t):
    """
    div operator a/b
    """
    return div_op(a_t, b_t)
    
def dual_wise_op_test():
    for op in [add, binary_crossentropy, categorical_crossentropy, sub, max, min, mul, div]:
        print(f'{op.__name__}()')
        for _ in range(10):    
            if op == categorical_crossentropy:
                shape_gen = [2]
            else:
                shape_gen = range(1, 5)
            
            for shape_len in shape_gen:
                try:                    
                    a_shape = tuple(np.random.randint( 8, size=(shape_len,) )+1)
                    
                    if op == categorical_crossentropy:
                        b_shape = a_shape
                    else:
                        if np.random.randint(2) == 0:
                            b_shape = tuple(a_shape[np.random.randint(len(a_shape)):])
                            b_shape = (1,) if len(b_shape) == 0 else b_shape
                        else:
                            b_shape = list(a_shape)
                            b_shape[ np.random.randint(len(b_shape)) ] = 1
                            b_shape = tuple(b_shape)

                        shapes = [a_shape, b_shape]
                        if np.random.randint(2) == 0:
                            shapes = shapes[::-1]
                        a_shape, b_shape = shapes

                    a_n = np.random.randint( 1, 2**8, size=a_shape ).astype(np.float32)
                    b_n = np.random.randint( 1, 2**8, size=b_shape ).astype(np.float32)
                    a_t = nn.Tensor_from_value(a_n)
                    b_t = nn.Tensor_from_value(b_n)
                    r_t = op(a_t, b_t)

                    r_n_grad = np.random.randint( 2**8, size=r_t.shape ).astype(np.float32)

                    a_t.get_grad().fill(1.0)
                    b_t.get_grad().fill(1.0)
                    nn.backward({r_t:r_n_grad}, grad_for_non_trainables=True)

                    if op == div:
                        # Test validness and gradient only for div
                        r_n = a_n / b_n
                        
                        if r_n.shape != r_t.shape:
                            raise Exception(f'shapes are not equal')
                        if np.abs(np.sum((np.ndarray.flatten(r_t.np() - r_n)))) > 1:
                            raise Exception(f'data is not equal')
   
                        info = nc.info.InfoBroadcast( nc.TensorShape(a_shape), nc.TensorShape(b_shape) )

                        a_n_grad = r_n_grad / b_n

                        axes = info.a_shape_reduction_axes
                        if axes.rank == 0:
                            a_n_grad = a_n_grad.reshape(a_n.shape)
                        else:
                            a_n_grad = a_n_grad.sum( tuple(axes), keepdims=True )

                        b_n_grad = r_n_grad * (-a_n/(b_n**2))

                        axes = info.b_shape_reduction_axes
                        if axes.rank == 0:
                            b_n_grad = b_n_grad.reshape(b_n.shape)
                        else:
                            b_n_grad = b_n_grad.sum( tuple(axes), keepdims=True )

                        if np.abs(np.sum((np.ndarray.flatten( a_t.get_grad().np() - 1.0 - a_n_grad)))) > 1:
                            raise Exception(f'grad A is not equal')
                        if np.abs(np.sum((np.ndarray.flatten( b_t.get_grad().np() - 1.0 - b_n_grad)))) > 1:
                            raise Exception(f'grad B is not equal')
                    else:
                        if not a_t.has_grad():
                            raise Exception(f'a_t has no grad')
                        if not b_t.has_grad():
                            raise Exception(f'b_t has no grad')
                        
                except:
                    raise Exception(f"""
op        : {op}
a_shape   : {a_shape}
b_shape   : {b_shape}
r_n_shape : {r_n.shape}
exception : {traceback.format_exc() }
""")

