import litenn.core as nc

class InfoBroadcast:
    """
    Broadcast info.

    arguments

        a_shape     TensorShape
        b_shape     TensorShape

    errors during the construction:

        ValueError

    Example:

     a_shape  = (   3,)
     b_shape  = (3, 1,)

                            #Example | comment
     a_br_shape             #(1, 3)    a_shape broadcasted to b_shape
     a_shape_reduction_axes #(0,)      a_shape_reduction_axes to make
                                         a_shape from output_shape
                                         
     a_tiles                #(3, 1)    a_tiles to make a_br_shape from output_shape

     b_br_shape             #(3, 1)    b_shape broadcasted to a_shape
     b_shape_reduction_axes #(1,)      b_shape_reduction_axes to make
                                         b_shape from output_shape
                                         
     b_tiles                #(1, 3)    b_tiles to make b_br_shape from output_shape

     output_shape           #(3, 3)    broadcasted together output_shape

    """

    __slots__ = ['a_br_shape', 'a_shape_reduction_axes', 'a_tiles', 'b_br_shape', 'b_shape_reduction_axes', 'b_tiles', 'output_shape']

    def __init__(self, a_shape, b_shape):
        a_br_shape = a_shape
        b_br_shape = b_shape

        # if shapes are inequal, broadcast with (1,)-s to left
        if a_br_shape.rank < b_br_shape.rank:
            diff = b_br_shape.rank-a_br_shape.rank
            a_br_shape = (1,)*diff + a_br_shape
        elif b_br_shape.rank < a_br_shape.rank:
            diff = a_br_shape.rank-b_br_shape.rank
            b_br_shape = (1,)*diff + b_br_shape

        # Now both shapes have the same rank
        rank = a_br_shape.rank

        a_tiles = [1,]*rank
        b_tiles = [1,]*rank

        a_shape_reduction_axes = []
        b_shape_reduction_axes = []

        # Validate axes and compute output_shape
        output_shape = []
        for axis in range(rank):
            a_axis_size = a_br_shape[axis]
            b_axis_size = b_br_shape[axis]

            if a_axis_size != b_axis_size:
                if  a_axis_size == 1 and b_axis_size != 1:
                    a_tiles[axis] = b_axis_size
                    a_shape_reduction_axes.append(axis)
                elif a_axis_size != 1 and b_axis_size == 1:
                    b_tiles[axis] = a_axis_size
                    b_shape_reduction_axes.append(axis)
                else:
                    # Unable to broadcast
                    raise ValueError(f'operands could not be broadcast together with shapes {a_shape} {b_shape}')

            output_shape.append( max(a_axis_size, b_axis_size) )

        self.output_shape = nc.TensorShape(output_shape)
        self.a_br_shape = a_br_shape
        self.a_shape_reduction_axes = nc.TensorAxes(a_shape_reduction_axes)
        self.a_tiles = tuple(a_tiles)
        self.b_br_shape = b_br_shape
        self.b_shape_reduction_axes = nc.TensorAxes(b_shape_reduction_axes)
        self.b_tiles = tuple(b_tiles)