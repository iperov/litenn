import traceback
import numpy as np
import litenn as nn
import litenn.core as nc
from litenn.core import CLKernelHelper as ph

class ElementWiseOpKernel:
    """
    Base class for kernels to use in element_wise_op()
    """
    def get_forward_kernel_text(self):
        """
        return kernel C code block for forward operation
        This block will be inserted to the complete OpenCL kernel code.

        available variables:
            I - input for forward
            O - store result of forward

        You can declare and use intermediate C variables.

        example code block for ReLU activation:

            return "O = I * (I >= 0);"
        """
        raise NotImplementedError()

    def get_backward_kernel_text(self):
        """
        return kernel C code for backward operation
        This block will be inserted to the complete OpenCL kernel code.

        available variables:
            I - input for forward
            O - result of forward
            dO - gradient of O
            dI - store result of backward for input I

        example code block for backward ReLU activation:

            return "dI = dO * (I >= 0);"
        """
        raise NotImplementedError()

    def get_op_name(self):
        raise NotImplementedError()

def element_wise_op(ElementWiseOpKernel_cls, ElementWiseOpKernel_args, input_t, output_t=None, is_add_to_output=False):
    """
    operator for ElementWiseOpKernel ops

    arguments

        ElementWiseOpKernel_cls     class of ElementWiseOpKernel

        ElementWiseOpKernel_args    args to construct ElementWiseOpKernel_cls

        output_t            compute result to this Tensor.
                            Tensor may be with different shape, but should match total size.
                            gradfn will not be set.

        is_add_to_output    add result to output_t if output_t is set.
    """
    is_add_to_output = False if output_t is None else is_add_to_output

    op = nc.Cacheton.get(_ElementWiseOp, ElementWiseOpKernel_cls, ElementWiseOpKernel_args, input_t.shape, is_add_to_output)

    if output_t is None:
        output_t = nn.Tensor ( op.output_shape )
        output_t._set_op_name(f'{op.kernel.get_op_name()}')
        output_t._assign_gradfn (input_t, lambda O_t, dO_t: input_1_gradfn(op, input_t, O_t, dO_t) )
    elif output_t.shape.size != op.output_shape.size:
        raise ValueError(f'output_t must have size {op.output_shape.size}')

    op.forward_krn.run(output_t, input_t)

    return output_t

def input_1_gradfn(op, input_t, O_t, dO_t):
    op.backward_krn.run(input_t.get_grad(), input_t, O_t, dO_t)


class _ElementWiseOp():
    def __init__(self, ElementWiseOpKernel_cls, ElementWiseOpKernel_args, input_shape, is_add_to_output):
        self.output_shape = input_shape
        self.kernel = ElementWiseOpKernel_cls(*ElementWiseOpKernel_args)

        self.forward_krn = nc.CLKernel(global_shape=(input_shape.size,), kernel_text=f"""
__kernel void impl(__global float* O_t, __global const float* I_t)
{{
    size_t idx = get_global_id(0);
    float I = I_t[idx];
    float O = 0.0;
    {self.kernel.get_forward_kernel_text()}
    O_t[idx] {'+=' if is_add_to_output else '='} O;
}}
""")

        self.backward_krn = nc.CLKernel(global_shape=(input_shape.size,), kernel_text=f"""
__kernel void impl(__global float* dI_t, __global const float* I_t, __global const float* O_t, __global const float* dO_t)
{{
    size_t idx = get_global_id(0);
    float I = I_t[idx];
    float O = O_t[idx];
    float dO = dO_t[idx];
    float dI = 0.0;
    {self.kernel.get_backward_kernel_text()}
    dI_t[idx] += dI;
}}
""")

class abs_kernel(ElementWiseOpKernel):
    def get_forward_kernel_text(self):  return f"O = fabs(I);"
    def get_backward_kernel_text(self): return f"dI = dO * ( I / fabs(I) );"
    def get_op_name(self): return f"abs"

def abs_op(input_t, output_t=None, is_add_to_output=False):
    return element_wise_op(abs_kernel, (), input_t, output_t=output_t, is_add_to_output=is_add_to_output)
def abs(input_t):
    return abs_op(input_t)

class add_const_kernel(ElementWiseOpKernel):
    def __init__(self, value): self.value = value
    def get_forward_kernel_text(self):  return f"O = I+({self.value});"
    def get_backward_kernel_text(self): return f"dI = dO;"
    def get_op_name(self): return f"add_const"
def add_const_op(input_t, value, output_t=None, is_add_to_output=False):
    return element_wise_op(add_const_kernel, (value,), input_t, output_t=output_t, is_add_to_output=is_add_to_output)
def add_const(input_t, value):
    return add_const_op(input_t, value)


class clip_kernel(ElementWiseOpKernel):
    def __init__(self, min_value, max_value,preserve_gradient=False):
        self.min_value, self.max_value, self.preserve_gradient = min_value, max_value, preserve_gradient
    def get_forward_kernel_text(self):  return f"O = I*(I>={self.min_value}&I<={self.max_value})+{self.min_value}*(I<{self.min_value})+{self.max_value}*(I>{self.max_value});"
    def get_backward_kernel_text(self):
        return  f"dI = dO;" if self.preserve_gradient else \
                f"dI = dO*(I>={self.min_value}&I<={self.max_value});"
    def get_op_name(self): return f"clip"
def clip_op(input_t, min_value, max_value, preserve_gradient=False, output_t=None, is_add_to_output=False):
    """
    Element-wise clip by min/max value
    
    arguments
    
            input_t     Tensor
            
            min_value   float
            
            max_value   float
            
            preserve_gradient(False)        
                        if False, gradient will be supressed
                        on values which are outside of range
    """
    if min_value > max_value:
        raise ValueError(f'{min_value} > {max_value}')
    return element_wise_op(clip_kernel, (min_value,max_value,preserve_gradient), input_t, output_t=output_t, is_add_to_output=is_add_to_output)
def clip(input_t, min_value, max_value, preserve_gradient=False):
    """
    Element-wise clip by min/max value
    
    arguments
    
            input_t     Tensor
            
            min_value   float
            
            max_value   float
            
            preserve_gradient(False)        
                        if False, gradient will be supressed
                        on values which are outside of range
    """
    return clip_op(input_t, min_value, max_value, preserve_gradient)

class cos_kernel(ElementWiseOpKernel):
    def get_forward_kernel_text(self):  return f"O = cos(I);"
    def get_backward_kernel_text(self): return f"dI = dO * -sin(I);"
    def get_op_name(self): return f"cos"
def cos_op(input_t, output_t=None, is_add_to_output=False):
    return element_wise_op(cos_kernel, (), input_t, output_t=output_t, is_add_to_output=is_add_to_output)
def cos(input_t):
    return cos_op(input_t)

def div_const_op(input_t, value, output_t=None, is_add_to_output=False):
    return mul_const_op(input_t, 1.0/value, output_t=output_t, is_add_to_output=is_add_to_output)
def div_const(input_t, value):
    return mul_const(input_t, 1.0/value )

class exp_kernel(ElementWiseOpKernel):
    def get_forward_kernel_text(self):  return f"O = exp(I);"
    def get_backward_kernel_text(self): return f"dI = dO * exp(I);"
    def get_op_name(self): return f"exp"
def exp_op(input_t, output_t=None, is_add_to_output=False):
    """
    Element-wise exponential of input_t.
    """
    return element_wise_op(exp_kernel, (), input_t, output_t=output_t, is_add_to_output=is_add_to_output)
def exp(input_t):
    """
    Element-wise exponential of input_t.
    """
    return exp_op(input_t)


class leaky_relu_kernel(ElementWiseOpKernel):
    def __init__(self, alpha=0.1): self.alpha = alpha
    def get_forward_kernel_text(self):  return f"O = I * (I >= 0) + {self.alpha} * I * (I < 0);"
    def get_backward_kernel_text(self): return f"dI = dO * ( (I >= 0) + {self.alpha} * (I < 0) );"
    def get_op_name(self): return f"leaky_relu({self.alpha})"

def leaky_relu_op(input_t, alpha, output_t=None, is_add_to_output=False):
    return element_wise_op(leaky_relu_kernel, (alpha,), input_t, output_t=output_t, is_add_to_output=is_add_to_output)
def leaky_relu(input_t, alpha=0.1):
    """
    leaky_relu operator

        alpha(0.1)     float
    """
    return leaky_relu_op(input_t, alpha)

class log_kernel(ElementWiseOpKernel):
    def get_forward_kernel_text(self):  return f"O = log(I);"
    def get_backward_kernel_text(self): return f"dI = dO * ( 1.0 / I );"
    def get_op_name(self): return f"log"
def log_op(input_t, output_t=None, is_add_to_output=False):
    """
    Element-wise natural logarithm.
    """
    return element_wise_op(log_kernel, (), input_t, output_t=output_t, is_add_to_output=is_add_to_output)
def log(input_t):
    """
    Element-wise natural logarithm.
    """
    return log_op(input_t)


class mul_const_kernel(ElementWiseOpKernel):
    def __init__(self, value): self.value = value
    def get_forward_kernel_text(self):  return f"O = I*({self.value});"
    def get_backward_kernel_text(self): return f"dI = dO*({self.value});"
    def get_op_name(self): return f"mul_const"
def mul_const_op(input_t, value, output_t=None, is_add_to_output=False):
    return element_wise_op(mul_const_kernel, (value,), input_t, output_t=output_t, is_add_to_output=is_add_to_output)
def mul_const(input_t, value):
    return mul_const_op(input_t, value)

class rdiv_const_kernel(ElementWiseOpKernel):
    def __init__(self, value): self.value = value
    def get_forward_kernel_text(self):  return f"O = ({self.value}) / I;"
    def get_backward_kernel_text(self): return f"dI = dO* ( -( ({self.value}) / (I*I))) ;"
    def get_op_name(self): return f"rdiv_const"
def rdiv_const_op(input_t, value, output_t=None, is_add_to_output=False):
    return element_wise_op(rdiv_const_kernel, (value,), input_t, output_t=output_t, is_add_to_output=is_add_to_output)
def rdiv_const(input_t, value):
    return rdiv_const_op(input_t, value)
    
class rsub_const_kernel(ElementWiseOpKernel):
    def __init__(self, value): self.value = value
    def get_forward_kernel_text(self):  return f"O = ({self.value})-I;"
    def get_backward_kernel_text(self): return f"dI = -dO;"
    def get_op_name(self): return f"rsub_const"
def rsub_const_op(input_t, value, output_t=None, is_add_to_output=False):
    return element_wise_op(rsub_const_kernel, (value,), input_t, output_t=output_t, is_add_to_output=is_add_to_output)
def rsub_const(input_t, value):
    return rsub_const_op(input_t, value)
      
class relu_kernel(ElementWiseOpKernel):
    def get_forward_kernel_text(self):  return "O = I * (I >= 0);"
    def get_backward_kernel_text(self): return "dI = dO * (I >= 0);"
    def get_op_name(self): return "relu"
def relu_op(input_t, output_t=None, is_add_to_output=False):
    """
    Element-wise relu operator.
    """
    return element_wise_op(relu_kernel, (), input_t, output_t=output_t, is_add_to_output=is_add_to_output)
def relu(input_t):
    """
    Element-wise relu operator.
    """
    return relu_op(input_t)

class sigmoid_kernel(ElementWiseOpKernel):
    def get_forward_kernel_text(self):  return f"O = 1.0 / (1.0 + exp(-I));"
    def get_backward_kernel_text(self): return f"dI = dO * ( O * (1.0 - O) );"
    def get_op_name(self): return f"sigmoid"
def sigmoid_op(input_t, output_t=None, is_add_to_output=False):
    """
    Element-wise sigmoid operator.
    """
    return element_wise_op(sigmoid_kernel, (), input_t, output_t=output_t, is_add_to_output=is_add_to_output)
def sigmoid(input_t):
    """
    Element-wise sigmoid operator.
    """
    return sigmoid_op(input_t)

class sin_kernel(ElementWiseOpKernel):
    def get_forward_kernel_text(self):  return f"O = sin(I);"
    def get_backward_kernel_text(self): return f"dI = dO * cos(I);"
    def get_op_name(self): return f"sin"
def sin_op(input_t, output_t=None, is_add_to_output=False):
    return element_wise_op(sin_kernel, (), input_t, output_t=output_t, is_add_to_output=is_add_to_output)
def sin(input_t):
    return sin_op(input_t)

def softmax(input_t, axis=-1):
    """
    Softmax operator.
    """
    e = exp(input_t)
    return e / e.sum (axis, keepdims=True)

class sqrt_kernel(ElementWiseOpKernel):
    def get_forward_kernel_text(self):  return f"O = sqrt(I);"
    def get_backward_kernel_text(self): return f"dI = dO * ( 1.0 / ( 2 * sqrt(I) ) );"
    def get_op_name(self): return f"sqrt"
def sqrt_op(input_t, output_t=None, is_add_to_output=False):
    """
    Element-wise sqrt operator.
    """
    return element_wise_op(sqrt_kernel, (), input_t, output_t=output_t, is_add_to_output=is_add_to_output)
def sqrt(input_t):
    """
    Element-wise sqrt operator.
    """
    return sqrt_op(input_t)
    
class square_kernel(ElementWiseOpKernel):
    def get_forward_kernel_text(self):  return f"O = I*I;"
    def get_backward_kernel_text(self): return f"dI = dO * 2 * I;"
    def get_op_name(self): return f"square"
def square_op(input_t, output_t=None, is_add_to_output=False):
    """
    Element-wise square operator.
    """
    return element_wise_op(square_kernel, (), input_t, output_t=output_t, is_add_to_output=is_add_to_output)
def square(input_t):
    """
    Element-wise square operator.
    """
    return square_op(input_t)

def sub_const_op(input_t, value, output_t=None, is_add_to_output=False):
    return add_const_op(input_t, -value, output_t=output_t, is_add_to_output=is_add_to_output)
def sub_const(input_t, value):
    return add_const(input_t, -value)

class tanh_kernel(ElementWiseOpKernel):
    def get_forward_kernel_text(self):  return f"O = 2.0 / (1.0 + exp(-2.0*I)) - 1.0;"
    def get_backward_kernel_text(self): return f"dI = dO * ( 1.0 - O * O );"
    def get_op_name(self): return f"tanh"
def tanh_op(input_t, output_t=None, is_add_to_output=False):
    """
    Element-wise tanh operator.
    """
    return element_wise_op(tanh_kernel, (), input_t, output_t=output_t, is_add_to_output=is_add_to_output)
def tanh(input_t):
    """
    Element-wise tanh operator.
    """
    return tanh_op(input_t)


def element_wise_op_test():
    add_params = { add_const  : [1.0],
                   clip       : [0.0, 1.0],
                   div_const  : [2.0],
                   mul_const  : [2.0],
                   leaky_relu : [0.1],
                   rdiv_const : [2.0],
                   rsub_const : [2.0],
                   sub_const  : [1.0],
                 }
    for op in [abs, add_const, clip, cos, div_const, exp, leaky_relu, log, mul_const, rdiv_const, rsub_const, relu, sigmoid, sin, softmax, sqrt, square, sub_const, tanh]:
      print(f'{op.__name__}()')
      for _ in range(10):
        for shape_len in range(1,3):
            try:
                shape = (np.random.randint( 8, size=(shape_len,) )+1).tolist()

                value_n = np.random.randint( 128, size=shape ).astype(np.float32)-64
                value_t = nn.Tensor_from_value(value_n)

                args = add_params.get(op, None)
                if args is None:
                    args = []

                result_t = op( *([value_t]+args) )
                result_t.backward(grad_for_non_trainables=True)
                
                if not value_t.has_grad():
                    raise Exception('No grad.')                

            except:
                raise Exception(f"""
shape       : {shape}
op          : {op.__name__}
args        : {args}
exception   : {traceback.format_exc()}
""")