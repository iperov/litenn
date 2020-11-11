import numpy as np
import litenn as nn

class Dropout(nn.Module):
    """
    Dropout module.

        Parameters

        rate       float    [0.0 .. 1.0)

    reference

    Dropout: A Simple Way to Prevent Neural Networks from Overfitting
    """
    def __init__(self, rate=0.5):
        super().__init__(saveables=[], trainables=[])

        if rate < 0 or rate >= 1.0:
            raise ValueError(f'rate {rate} must be in range [0 .. 1.0)')
        self.rate = rate

    def forward(self, x, **kwargs):
        if self.is_training():
            x = nn.dropout(x, self.rate)
        return x

    def __str__(self): return f"{self.__class__.__name__} : rate:{self.rate} "
    def __repr__(self): return self.__str__()

def Dropout_test():
    module = Dropout(0.3)

    module.set_training(True)
    x = nn.Tensor( (2,4,4,4) )
    y = module(x)
    y.backward(grad_for_non_trainables=True)

    if not x.has_grad():
        raise Exception('x has no grad')

    module.set_training(False)
    x = nn.Tensor( (2,4,4,4) )
    y = module(x)
    y.backward(grad_for_non_trainables=True)
