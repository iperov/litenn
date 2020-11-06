from collections import Iterable
from .TensorAxes import TensorAxes

class TensorShape(Iterable):
    """
    Constructs valid shape from user argument

    arguments

     shape      TensorShape
                Iterable

    can raise ValueError during the construction
    """

    __slots__ = ['shape','size','rank']

    def __init__(self, shape):
        if isinstance(shape, TensorShape):
            self.shape = shape.shape
            self.size = shape.size
            self.rank = shape.rank
        else:
            if isinstance(shape, (int,float) ):
                shape = (int(shape),)
                
            if isinstance(shape, Iterable):
                size = 1
                valid_shape = []
                for x in shape:
                    if x is None:
                        raise ValueError(f'Incorrent value {x} in shape {shape}')
                    x = int(x)
                    if x < 1:
                        raise ValueError(f'Incorrent value {x} in shape {shape}')
                    valid_shape.append(x)
                    size *= x # Faster than np.prod()

                self.shape = tuple(valid_shape)
                self.rank = len(self.shape)
                if self.rank == 0:
                    # Force (1,) shape for scalar shape
                    self.rank = 1
                    self.shape = (1,)
                self.size = size
            else:
                raise ValueError('Invalid type to create TensorShape')

    def axes_arange(self):
        """
        Returns tuple of axes arange.

         Example (0,1,2) for rank 3
        """
        return TensorAxes(range(self.rank))

    def transpose_by_axes(self, axes):
        """
        Same as TensorShape[axes]

        Returns TensorShape transposed by axes.

         axes       TensorAxes
                    Iterable(list,tuple,set,generator)
        """
        return TensorShape(self.shape[axis] for axis in TensorAxes(axes) )

    def __hash__(self): return self.shape.__hash__()
    def __eq__(self, other):
        if isinstance(other, TensorShape):
            return self.shape == other.shape
        elif isinstance(other, Iterable):
            return self.shape == tuple(other)
        return False
    def __iter__(self): return self.shape.__iter__()
    def __len__(self): return len(self.shape)
    def __getitem__(self,key):
        if isinstance(key, Iterable):
            if isinstance(key, TensorAxes):
                if key.is_none_axes():
                    return self
            return self.transpose_by_axes(key)
        elif isinstance(key, slice):
            return TensorShape(self.shape[key])

        return self.shape[key]

    def __radd__(self, o):
        if isinstance(o, Iterable):
            return TensorShape( tuple(o) + self.shape)
        else:
            raise ValueError(f'unable to use type {o.__class__} in TensorShape append')
    def __add__(self, o):
        if isinstance(o, Iterable):
            return TensorShape( self.shape + tuple(o) )
        else:
            raise ValueError(f'unable to use type {o.__class__} in TensorShape append')


    def __str__(self):  return str(self.shape)
    def __repr__(self): return 'TensorShape' + self.__str__()

