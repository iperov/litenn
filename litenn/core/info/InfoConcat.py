import litenn.core as nc

class InfoConcat:
    """
    Concat info

    arguments

        shapes      Iterable of Tensors

    errors during the construction:

        ValueError

    result

        .output_shape   TensorShape

        .axis           Int     fixed axis argument

        .axis_sizes     List[Int]   axis sizes for every shape in shapes

        .axis_offsets   List[Int]   axis offset in output_shape
                                    for every shape in shapes
    """

    __slots__ = ['output_shape', 'axis', 'axis_sizes',  'axis_offsets']

    def __init__(self, shapes, axis):
        shapes = tuple(shapes)

        if len(shapes) == 0:
            raise ValueError('shapes is empty')


        shape = shapes[0]

        if axis < 0:
            axis = shape.rank + axis
        if axis < 0 or axis >= shape.rank:
            raise ValueError(f'Wrong axis {axis}')

        fixed_shapes = [ tuple(a for i,a in enumerate(shape) if i != axis) for shape in shapes ]
        req_shape = fixed_shapes[0]
        if not all(shape == req_shape for shape in fixed_shapes[1:]):
            raise ValueError(f'All shapes must match shape {tuple(a if i != axis else "*" for i,a in enumerate(shape))}')

        axis_sizes = [ shape[axis] for shape in shapes ]

        axis_offset = 0
        axis_offsets = []
        for axis_size in axis_sizes:
            axis_offsets.append(axis_offset)
            axis_offset += axis_size

        self.output_shape = nc.TensorShape( tuple(shape)[0:axis] + (sum(axis_sizes),) + tuple(shape)[axis+1:] )
        self.axis = axis
        self.axis_sizes = axis_sizes
        self.axis_offsets = tuple(axis_offsets)



