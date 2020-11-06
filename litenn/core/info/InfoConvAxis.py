from collections import Iterable

class InfoConvAxis:
    """
    Convolution info for single axis. 
    
    can raise an errors during the construction
    
     axis_size      int
     
     kernel_size    int
     
     stride         int
     
     dilation       int
     
     padding     'valid'         no padding
                 'same'          output size will be the same or divided by stride
                 int             padding value for all sides
                 tuple of24 ints paddings for left,right sides

    """
    
    __slots__ = [
        'PADL',  # Padding for left size
        'PADR',  # Padding for right size
        
        'O_axis_size',   # Convolved axis size
        
        'OT_axis_size',  # Convolved transposed axis size.
                         # only for 'valid' and 'same' padding,
                         # otherwise == None
    ]
    
    def __init__(self, axis_size, kernel_size, stride, dilation, padding):        
        # Effective kernel sizes with dilation
        EKS = (kernel_size-1)*dilation + 1

        # Determine pad size of sides
        OT_axis_size = None
        if padding == 'valid':
            PADL = PADR = 0
            OT_axis_size =  axis_size * stride + max(EKS - stride, 0)
        elif padding == 'same':
            PADL = int(math.floor((EKS - 1)/2))
            PADR = int(math.ceil((EKS - 1)/2))
            OT_axis_size = axis_size * stride      
        elif isinstance(padding, int):
            PADL = PADR = padding
        elif isinstance(padding, Iterable):
            padding = tuple(int(x) for x in padding)
            if len(padding) != 2:
                raise ValueError("Invalid paddings list length.")
            PADL, PADR = padding
        else:
            raise ValueError("Invalid padding value.")
        
        self.PADL = PADL
        self.PADR = PADR
            
        self.O_axis_size = max(1, int((axis_size + PADL + PADR - EKS) / stride + 1) )
        self.OT_axis_size = OT_axis_size
                                    