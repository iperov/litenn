import numpy as np
import litenn as nn
import litenn.core as nc

class BlurPool(nn.Module):
    """
    BlurPool operator.
    
    works with any channels tensor.
    
    arguments
    
     kernel_size(3) int
 
     stride(2)      int
     
     dilation(1)    int
     
     padding(same)   'valid'         no padding
                     'same'          output size will be the same 
                                     or divided by stride
                     int             padding value for all sides
                     Iterable of 4 ints 
                                paddings for left,top,right,bottom sides
                                
    reference
    
    Making Convolutional Networks Shift-Invariant Again
    https://arxiv.org/pdf/1904.11486.pdf
    """
    def __init__(self, kernel_size=3, stride=2, dilation=1, padding='same' ):
        super().__init__(saveables=[], trainables=[])
        
        self.stride = stride
        self.dilation = dilation
        self.padding = padding

        if(kernel_size==1):
            a = np.array([1.,])
        elif(kernel_size==2):
            a = np.array([1., 1.])
        elif(kernel_size==3):
            a = np.array([1., 2., 1.])
        elif(kernel_size==4):
            a = np.array([1., 3., 3., 1.])
        elif(kernel_size==5):
            a = np.array([1., 4., 6., 4., 1.])
        elif(kernel_size==6):
            a = np.array([1., 5., 10., 10., 5., 1.])
        elif(kernel_size==7):
            a = np.array([1., 6., 15., 20., 15., 6., 1.])

        a = a[:,None]*a[None,:]
        a = a / np.sum(a)
        a = a[None,:,:]
        self.a = nn.Tensor_from_value(a)


    def forward(self, x):
        n,c,h,w = x.shape
        
        a = nn.tile (self.a, (c,1,1) )
        
        x = nn.depthwise_conv2D(x, a, self.stride, self.dilation, self.padding)
        return x
        
def BlurPool_test():
    module = BlurPool()        
    x = nn.Tensor( (2,3,64,64) )
    y = module(x)
    y.backward(grad_for_non_trainables=True)
    
    if not x.has_grad():
        raise Exception('x has no grad')