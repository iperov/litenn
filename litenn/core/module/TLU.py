import litenn as nn

class TLU(nn.Module):
    """
    Thresholded Linear Unit

    arguments

        in_ch       input channels

    reference

    Filter Response Normalization Layer:
    Eliminating Batch Dependence in the Training of Deep Neural Networks
    https://arxiv.org/pdf/1911.09737.pdf
    """
    def __init__(self, in_ch, dtype=None, **kwargs):
        self.in_ch = in_ch
        self.tau = nn.Tensor( (in_ch,), init=nn.initializer.Scalar(0.0) )

        super().__init__(saveables=['tau'])

    def forward(self, x):
        return nn.max(x, self.tau.reshape( (1,-1,1,1)) )

def TLU_test():
    in_ch = 3
    module = TLU(in_ch)
    x = nn.Tensor( (2,in_ch,64,64) )
    y = module(x)
    y.backward(grad_for_non_trainables=True)

    if not x.has_grad():
        raise Exception(f'x has no grad')