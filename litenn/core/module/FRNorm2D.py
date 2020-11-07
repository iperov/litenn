import litenn as nn


class FRNorm2D(nn.Module):
    """
    Filter Response Normalization.
    
    should be used with nn.TLU() activator
    
    arguments
    
        in_ch       input channels
    
    reference 
    
    Filter Response Normalization Layer: 
    Eliminating Batch Dependence in the Training of Deep Neural Networks
    https://arxiv.org/pdf/1911.09737.pdf
    """
    def __init__(self, in_ch):
        self.in_ch = in_ch

        self.weight      = nn.Tensor ( (in_ch,), init=nn.initializer.Scalar(1.0) )
        self.bias        = nn.Tensor ( (in_ch,), init=nn.initializer.Scalar(0.0) )
        self.eps         = nn.Tensor ( (1,),  init=nn.initializer.Scalar(1e-6) )

    def forward(self, x):
        nu2 = nn.square(x).mean( (-2,-1), keepdims=True)
        
        x = x * ( 1.0/nn.sqrt(nu2 + nn.abs(self.eps) ) )

        return x*self.weight.reshape ( (1,-1,1,1) ) \
               + self.bias.reshape   ( (1,-1,1,1) )
        
def FRNorm2D_test():
    in_ch = 3
    module = FRNorm2D(in_ch)        
    x = nn.Tensor( (2,in_ch,64,64) )
    y = module(x)
    y.backward(grad_for_non_trainables=True)
    
    if not x.has_grad():
        raise Exception(f'x has no grad')