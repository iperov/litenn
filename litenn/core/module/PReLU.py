import litenn as nn
import litenn.core as nc

class PReLU (nn.Module):
    """
    Parametric ReLU
    
    arguments
    
        in_ch       int     number of channels
        
        axis(None)  int     axis of channels
                            if None:
                            =1 if shape.rank==2
                            =1 if shape.rank==4
                            
    reference
    
    Delving Deep into Rectifiers: Surpassing Human-Level Performance on ImageNet Classification
    https://arxiv.org/abs/1502.01852
    """
    
    def __init__(self, in_ch, axis=None):
        self.in_ch = in_ch
        self.axis = axis
        self.alpha = nn.Tensor ( (in_ch,), init=nn.initializer.Scalar(0.0) )
        super().__init__(saveables=['alpha'])
        
    def forward(self, x):
        shape = x.shape
        
        alpha = self.alpha
        
        if self.axis is not None:
            new_shape = [1]*shape.rank
            new_shape[self.axis] = -1
            alpha = alpha.reshape (new_shape)
        else:
            if shape.rank == 2:
                alpha = alpha.reshape ( (1,-1) )
            elif shape.rank == 4:
                alpha = alpha.reshape ( (1,-1,1,1) )
            else:
                raise ValueError(f'Unsupported shape rank {shape.rank}, should be 2 or 4, or specify the axis directly.')

        return _prelu(x, alpha)

class _prelu_kernel(nc.op.DualWiseOpKernel):
    def get_forward_kernel_text(self):  return f"O = A*(A>=0) + B*A*(A<0);"
    def get_backward_A_kernel_text(self): return f"dA = dO*(A>=0) + dO*B*(A<0);"
    def get_backward_B_kernel_text(self): return f"dB = dO*A*(A<0);"
    def get_op_name(self): return f"_prelu"
def _prelu(a_t, b_t):
    return nc.op.dual_wise_op(_prelu_kernel, (), a_t, b_t)
    
def PReLU_test():
    in_ch = 3
    module = PReLU(in_ch)
    x = nn.Tensor( (2,in_ch,64,64) )
    y = module(x)
    y.backward(grad_for_non_trainables=True)

    if not x.has_grad():
        raise Exception(f'x has no grad')