import litenn.core as nc


class InfoStack:
    """
    Stack info   
    
    arguments
    
        shape           TensorShape
        
        axis            Int
        
        stack_count     Int
      
    errors during the construction:
    
        ValueError
        
    result:
    
        .output_shape       TensorShape
        
        .axis     Int       positive axis argument    
    """
    
    __slots__ = ['output_shape', 'axis']
    
    def __init__(self, shape, axis, stack_count):
        if axis < 0:
            axis = shape.rank + 1 + axis            
        if axis < 0 or axis > shape.rank:
            raise ValueError(f'Wrong axis {axis}')
        
        if stack_count <= 0:
            raise ValueError(f'Invalid stack_count {stack_count}')
        
        self.output_shape = nc.TensorShape( tuple(shape)[0:axis] + (stack_count,) + tuple(shape)[axis:] )
        self.axis = axis