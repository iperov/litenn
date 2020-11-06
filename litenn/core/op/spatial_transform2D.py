import traceback
import math
import numpy as np

import litenn as nn
import litenn.core as nc
from litenn.core import CLKernelHelper as ph
    

def spatial_transform2D (input_t, coords_t, grad_to_coords=True):
    """
    spatial_transform2D operator
    
    Transforms input_t in spatial axes using coords_t.

    arguments

        input_t     Tensor(NCHW)

        coords_t    Tensor(NCHWD)
                    N is 1 or input_t.N
                    C is 1 or input_t.C
                    H - output height
                    W - output width
                    D is (2)[x,y] coords

        grad_to_coords(True)

                if True, broadcasts coords_t to input_t for backprop
                if you use spatial_transform2D for resize only,
                you don't need backprop to coords_t

    Reference:

    Spatial Transformer Networks https://arxiv.org/abs/1506.02025
    """

    op = nc.Cacheton.get(_SpatialTransform2DOp, input_t.shape, coords_t.shape)

    output_t = nn.Tensor( op.output_shape )
    output_t._set_op_name('spatial_transform2D')
    output_t._assign_gradfn (input_t,  lambda O_t, dO_t: spatial_transform2D_dI_gradfn(op, input_t, coords_t, dO_t) )

    if grad_to_coords:
        dK_coords_t = coords_t

        diff_rank = 5-dK_coords_t.shape.rank

        if diff_rank != 0:
            dK_coords_t = dK_coords_t.reshape( diff_rank*(1,) + dK_coords_t.shape )

        if op.coords_N_tile != 1 or op.coords_C_tile != 1:
            dK_coords_t = nc.op.tile(dK_coords_t, (op.coords_N_tile,op.coords_C_tile,1,1,1) )

        output_t._assign_gradfn (dK_coords_t, lambda O_t, dO_t: spatial_transform2D_dK_gradfn(op, input_t, dK_coords_t, dO_t) )

    op.O_forward_krn.run(output_t, input_t, coords_t)

    return output_t

def spatial_transform2D_dI_gradfn(op, input_t, coords_t, dO_t):
    op.dI_krn.run(input_t.get_grad(), coords_t, dO_t)

def spatial_transform2D_dK_gradfn(op, input_t, coords_t, dO_t):
    op.dK_krn.run(coords_t.get_grad(), input_t, coords_t, dO_t)


class _SpatialTransform2DOp():
    def __init__(self, input_shape : nc.TensorShape, coords_shape : nc.TensorShape):
        N,IC,IH,IW = input_shape

        if coords_shape.rank not in [3,4,5]:
            raise ValueError(f'Coords shape rank must be 3(HWD) or 4(CHWD) or 5(NCHWD)')
        KN,KC = 1,1

        if coords_shape.rank == 5:
            KN,KC,KH,KW,KD = coords_shape
        elif coords_shape.rank == 4:
            KC,KH,KW,KD = coords_shape
        elif coords_shape.rank == 3:
            KH,KW,KD = coords_shape

        self.coords_N_tile = 1
        self.coords_C_tile = 1

        if KN != N:
            if KN == 1:
                self.coords_N_tile = N
            else:
                raise ValueError(f'Coords output batch {KN} does not match tensor input batch {N}.')

        if KC != IC:
            if KC == 1:
                self.coords_C_tile = IC
            else:
                raise ValueError(f'Coords output channels {KC} does not match tensor input channels {IC}.')

        if KD != 2:
            raise ValueError(f'Coords D {KD} channels must be == 2 (x,y)')

        self.output_shape = output_shape = nc.TensorShape( (N, IC, KH, KW) )


        common_kernel_text = f"""
{ph.define_axes_accessor('I', input_shape, 'NCHW')}
{ph.define_axes_accessor('O', output_shape, 'NCHW')}
"""
        self.O_forward_krn = nc.CLKernel(global_shape=(output_shape.size,), kernel_text=f"""
{common_kernel_text}
{ph.define_axes_accessor('K', (KN,KC,KH,KW), 'NCHW')}
__kernel void impl(__global float* O, __global const float* I, __global const float2* K)
{{
    size_t gid = get_global_id(0);
    {ph.axes_idxs_from_var('o', 'NCHW', 'gid')}

    float v = 0.0;

    float2 xys = K[K_idx_mod(on,oc,oh,ow)];

    for (int ih=0; ih < IH; ++ih)
    {{
        float ys_mod = max(0.0, 1.0-fabs(xys.y-ih));
        if (ys_mod != 0.0)
        for (int iw=0; iw < IW; ++iw)
        {{
            float xs_mod = max(0.0, 1.0-fabs(xys.x-iw));
            if (xs_mod != 0.0)
                v += xs_mod*ys_mod*I[I_idx(on,oc,ih,iw)];
        }}
    }}

    O[gid] = v;
}}
""")

        self.dI_krn = nc.CLKernel(global_shape=(input_shape.size,), kernel_text=f"""
{common_kernel_text}
{ph.define_axes_accessor('K', (KN,KC,KH,KW), 'NCHW')}
__kernel void impl(__global float* dI, __global const float2* K, __global float* dO)
{{
    size_t gid = get_global_id(0);
    {ph.axes_idxs_from_var('i', 'NCHW', 'gid')}

    float v = 0.0;

    for (int oh=0; oh < OH; ++oh)
    for (int ow=0; ow < OW; ++ow)
    {{
        float2 xys = K[K_idx_mod(in,ic,oh,ow)];

        float ys_mod = max(0.0, 1.0-fabs(xys.y-ih));
        if (ys_mod != 0.0)
        {{
            float xs_mod = max(0.0, 1.0-fabs(xys.x-iw));
            if (xs_mod != 0.0)
                v += xs_mod*ys_mod*dO[O_idx(in,ic,oh,ow)];
        }}
    }}

    dI[gid] += v;
}}
""")

        self.dK_krn = nc.CLKernel(global_shape=( N*IC*KH*KW,), kernel_text=f"""
{common_kernel_text}
{ph.define_axes_accessor('K', (N,IC,KH,KW), 'NCHW')}
__kernel void impl(__global float2* dK, __global const float* I, __global const float2* K, __global float* dO)
{{
    size_t gid = get_global_id(0);
    {ph.axes_idxs_from_var('k', 'NCHW', 'gid')}

    float dk_x = 0.0;
    float dk_y = 0.0;

    float2 xys = K[gid];

    for (int ih=0; ih < IH; ++ih)
    {{
        {{
            float ys_mod = max(0.0, 1.0-fabs(xys.y-ih));

            if (ys_mod != 0.0)
            for (int iw=0; iw < IW; ++iw)
            if ( fabs(iw-xys.x) < 1.0 )
            {{
                float xs_mod = 1.0*(iw >= xys.x)-1.0*(iw < xys.x);
                dk_x += xs_mod*ys_mod*I[I_idx(kn,kc,ih,iw)] * dO[O_idx(kn,kc,kh,kw)];
            }}
        }}

        if (fabs(ih-xys.y) < 1.0)
        {{
            float ys_mod = 1.0*(ih >= xys.y)-1.0*(ih < xys.y);
            for (int iw=0; iw < IW; ++iw)
            {{
                float xs_mod = max(0.0, 1.0-fabs(xys.x-iw));
                if (xs_mod != 0.0)
                    dk_y += xs_mod*ys_mod*I[I_idx(kn,kc,ih,iw)] * dO[O_idx(kn,kc,kh,kw)];
            }}
        }}
    }}
    dK[gid] += (float2)(dk_x, dk_y);
}}
""")