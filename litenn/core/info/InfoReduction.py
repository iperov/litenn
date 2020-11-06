import litenn.core as nc

class InfoReduction:
    """
    Reduction info
    
    arguments
    
        shape       TensorShape
        
        axes        TensorAxes
        
        keepdims    bool
        
    can raise ValueError, TypeError during the construction
    """
    
    __slots__ = [
        'reduction_axes',        # sorted reduction TensorAxes
        'output_axes',           # remain TensorAxes after reduction
        'output_shape',          # result TensorShape of reduction 
        'output_shape_keepdims', # result TensorShape of reduction with keepdims
    ]
    
    def __init__(self, shape, axes, keepdims):
        shape_axes = shape.axes_arange()
        
        if axes.is_none_axes():
            axes = shape_axes        
        
        # Check correctness of axes
        for axis in axes:
            if axis not in shape_axes:
                raise ValueError(f'Wrong axis {axis} not in {shape_axes}')
    
        self.reduction_axes = reduction_axes = axes.sorted()
        
        # Output axes. Remove axes from shape_axes
        self.output_axes = output_axes = shape_axes - axes
        
        if output_axes.is_none_axes():
            output_shape = nc.TensorShape( (1,) )
        else:
            output_shape = shape[output_axes]        
        
        self.output_shape = output_shape
        self.output_shape_keepdims = nc.TensorShape( 1 if axis in reduction_axes else shape[axis] for axis in range(shape.rank))
        
        if keepdims:
            self.output_shape = self.output_shape_keepdims