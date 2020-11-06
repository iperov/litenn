import numpy as np

import litenn as nn
import litenn.core as nc
from litenn.core import CLKernelHelper as ph

def spatial_affine_transform2D (input_t, affine_t, output_size=None):
    """
    arguments

        input_t     Tensor(NCHW)

        affine_t    Tensor(N,2,3)
                    affine matrix

                    example of identity affine matrix
                    [1,0,0],
                    [0,1,0]

        output_size(None)

                    tuple of 2 ints (HW)
                    of output size
                    if None , size will not be changed

    Reference:

    Spatial Transformer Networks https://arxiv.org/abs/1506.02025
    """

    op = nc.Cacheton.get(_SpatialAffineTransform2DOp, input_t.shape, affine_t.shape, output_size)

    affine_t = affine_t.transpose( (0,2,1) )

    coords = nn.Tensor(op.coords_shape, init=op.coords_init ).reshape(op.coords_reshape)
    coords = nc.op.matmul(coords, affine_t).reshape(op.coords_affined_shape)

    output_t = nc.op.spatial_transform2D(input_t, coords)
    return output_t


class _SpatialAffineTransform2DOp():
    def __init__(self, input_shape : nc.TensorShape, affine_shape : nc.TensorShape, output_size):
        if input_shape.rank != 4:
            raise ValueError('input_shape must be rank 4 (NCHW)')

        N,IC,IH,IW = input_shape

        if affine_shape.rank != 3:
            raise ValueError('affine_shape must be rank 3')

        AN, AH, AW = affine_shape
        self.aff_tile = 0
        if AN != N:
            raise ValueError(f'affine matrix batch must == {AN}')

        if AH != 2:
            raise ValueError('affine matrix height must == 2')
        if AW != 3:
            raise ValueError('affine matrix width  must == 3')

        if output_size is not None:
            OH,OW = output_size
        else:
            OH,OW = IH,IW

        self.coords_shape = nc.TensorShape( (N,IC,OH,OW,3)  )
        self.coords_init  = nc.initializer.CoordsArange(0,IH-1,0,IW-1)

        self.coords_reshape = nc.TensorShape( (N,IC*OH*OW,3) )
        self.coords_affined_shape = nc.TensorShape( (N,IC,OH,OW,2) )

        self.output_shape = output_shape = nc.TensorShape( (N,IC,OH,OW) )
