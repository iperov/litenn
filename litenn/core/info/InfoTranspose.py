import litenn.core as nc

class InfoTranspose:
    """
    TransposeInfo

    arguments

        shape           TensorShape

        axes_order      TensorAxes

    errors during the construction:

        ValueError

    result

        .output_shape   TensorShape

        .no_changes     bool       transpose changes nothing

    """

    __slots__ = ['no_changes', 'output_shape']

    def __init__(self, shape, axes_order):
        if shape.rank != axes_order.rank:
            raise ValueError('axes must match the shape')

        # Axes order changes nothing?
        self.output_shape = shape[axes_order]
        self.no_changes = axes_order == shape.axes_arange()



