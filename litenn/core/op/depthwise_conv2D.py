import traceback
import math
import numpy as np

import litenn as nn
import litenn.core as nc
from litenn.core import CLKernelHelper as ph

def depthwise_conv2D (input_t, kernel_t, stride=1, dilation=1, padding='same'):
    """
    Depthwise Conv2D operator.
   
     input_t     Tensor          shape must be 
                                 (batch, in_ch,  height,  width)
     
     kernel_t    Tensor          shape must be 
                                 (in_ch,k_height,k_width)
 
     stride(1)       int
     
     dilation(1)     int
     
     padding(same)   'valid'         no padding
                     'same'          output size will be the same 
                                     or divided by stride
                     int             padding value for all sides
                     Iterable of 4 ints 
                                paddings for left,top,right,bottom sides
    """
    
    op = nc.Cacheton.get(_DepthwiseConv2DOp, input_t.shape, kernel_t.shape, int(stride), int(dilation), padding)

    output_t = nn.Tensor( op.output_shape )
    output_t._set_op_name('depthwise_conv2D')
    output_t._assign_gradfn (input_t,  lambda O_t, dO_t: conv2D_dI_gradfn(op, input_t, kernel_t, dO_t) )
    output_t._assign_gradfn (kernel_t, lambda O_t, dO_t: conv2D_dK_gradfn(op, input_t, kernel_t, dO_t) )

    op.O_depthwise_krn.run(output_t, input_t, kernel_t)

    return output_t

def conv2D_dI_gradfn(op, input_t, kernel_t, dO_t):
    op.dI_depthwise_krn.run(input_t.get_grad(), kernel_t, dO_t)

def conv2D_dK_gradfn(op, input_t, kernel_t, dO_t):

    dK = op.im2col(input_t).reshape(op.KI_KH_KW_NxOHxOW) * \
            dO_t.transpose((1,0,2,3)).reshape(op.OC_1_1_NxOHxOW)

    nc.op.reduce_sum (dK, -1, output_t=kernel_t.get_grad(), is_add_to_output=True)

class _DepthwiseConv2DOp():
    def __init__(self, input_shape : nc.TensorShape, kernel_shape : nc.TensorShape, stride, dilation, padding):
        if kernel_shape.rank != 3:
            raise ValueError(f'Kernel shape rank must be == 3')   
        N,IC,IH,IW = input_shape
        KI,KH,KW = kernel_shape      
        if KI != IC:
            raise ValueError(f'Kernel input channels {KI} does not match tensor input channels {IC}.')
        
        ci = nc.info.InfoConv2D(IH, IW, KH, KW, stride, dilation, padding)
        OC, OH, OW = IC, ci.OH, ci.OW
        self.output_shape = output_shape = nc.TensorShape( (N, OC, OH, OW) )

        self.OC_1_1_NxOHxOW = (OC,1,1,N*OH*OW)
        self.KI_KH_KW_NxOHxOW = (KI,KH,KW,N*OH*OW)

        common_kernel_text = f"""
{ph.define_axes_accessor('I', input_shape, 'NCHW')}
{ph.define_axes_accessor('O', output_shape, 'NCHW')}
{ph.define_axes_accessor('K', kernel_shape, 'IHW')}
#define PADL {ci.PADL}
#define PADT {ci.PADT}

#define STRIDE {stride}
#define DILATION {dilation}
"""
        self.O_depthwise_krn = nc.CLKernel(global_shape=(output_shape.size,), kernel_text=f"""
{common_kernel_text}
__kernel void impl(__global float* O, __global const float* I, __global const float* K)
{{
    size_t gid = get_global_id(0);
    {ph.axes_idxs_from_var('o', 'NCHW', 'gid')}

    float v = 0.0;
    for (int kh=0; kh<KH; ++kh)
    {{
        int ih = -PADT + kh*DILATION + oh*STRIDE;
        if (ih >= 0 & ih < IH)
            for (int kw=0; kw<KW; ++kw)
            {{
                int iw = -PADL + kw*DILATION + ow*STRIDE;
                if (iw >= 0 & iw < IW)
                    v += I[I_idx(on,oc,ih,iw)]*K[K_idx(oc,kh,kw)];
            }}
    }}
    O[gid] = v;
}}
""")

        self.dI_depthwise_krn = nc.CLKernel(global_shape=(input_shape.size,), kernel_text=f"""
{common_kernel_text}
__kernel void impl(__global float* dI, __global const float* K, __global const float* dO)
{{
    size_t gid = get_global_id(0);
    {ph.axes_idxs_from_var('i', 'NCHW', 'gid')}
    float v = 0.0;
    for (int kh=0; kh<KH; ++kh)
    {{
        int oh = (PADT + ih - kh*DILATION ) / STRIDE;
        if (oh >= 0 & oh < OH)
            for (int kw=0; kw<KW; ++kw)
            {{
                int ow = (PADL + iw - kw*DILATION ) / STRIDE;
                if (ow >= 0 & ow < OW
                    & iw == (-PADL + kw*DILATION + ow*STRIDE)
                    & ih == (-PADT + kh*DILATION + oh*STRIDE) )
                        v += dO[O_idx(in,ic,oh,ow)]*K[K_idx(ic,kh,kw)];
            }}
    }}
    dI[gid] += v;
}}
""")

        self.im2col  = lambda x: nc.op.unfold2D(x, N, IC, IH, IW, OH, OW, KH, KW, ci.PADL, ci.PADT, dilation, stride, 'CJI_NHW', is_transpose=False)
        

def depthwise_conv2d_test():
    for padding in ['same','valid',0,1,2]:
        for dilation in [1,2]:
          for stride in [1,2,3]:
            for ks in [1,3,5,7]:
              for n in [1,4]:
                for ic in [1,2,4]:
                    for ih,iw in zip(*[[4,8,16]]*2):                        
                        if padding == 'valid' and iw < ks:
                            continue
                        try:
                            input_shape  = (n, ic, ih, iw)
                            kernel_shape = (ic, ks, ks)

                            input_n  = np.random.randint( 2**4, size=input_shape ).astype(np.float32)
                            kernel_n = np.random.randint( 2**4, size=kernel_shape ).astype(np.float32)

                            input_t  = nn.Tensor_from_value(input_n)
                            kernel_t = nn.Tensor_from_value(kernel_n)

                            conved_t = nn.depthwise_conv2D(input_t, kernel_t, stride=stride, dilation=dilation, padding=padding)
                            conved_n_grad = np.random.randint( 2**4, size=conved_t.shape).astype(np.float32)
                            conved_n, dI_val, dK_val = _numpy_depthwise_conv2d(input_n, kernel_n, conved_n_grad, STRIDE=stride, DILATION=dilation, padding=padding)

                            if conved_n.shape != conved_t.shape:
                                raise Exception(f'shape is not equal')

                            if not all ( np.ndarray.flatten( conved_t.np() == conved_n) ):
                                raise Exception(f'data is not equal')

                            input_t.get_grad().fill(1.0)
                            kernel_t.get_grad().fill(1.0)
                            nn.backward( {conved_t:conved_n_grad}, grad_for_non_trainables=True )

                            if not all ( np.ndarray.flatten( (input_t.get_grad().np()-1.0) == dI_val )):
                                raise Exception(f'dI not equal')

                            if not all ( np.ndarray.flatten( (kernel_t.get_grad().np()-1.0) == dK_val )):
                                raise Exception(f'dK not equal')
                        except:
                            raise Exception(f"""
input_shape   : {input_shape}
kernel_shape  : {kernel_shape}
padding       : {padding}
stride        : {stride}
dilation      : {dilation}
conved_n.shape : {conved_n.shape}
conved_t.shape : {conved_t.shape}
{traceback.format_exc()}
""")


def _numpy_depthwise_conv2d(input_n, kernel_n, conved_n_grad, STRIDE=1, DILATION=1, padding='same'):
    N, IC, IH, IW = input_n.shape
    KI, KH, KW = kernel_n.shape
    
    ci = nc.info.InfoConv2D(IH, IW, KH, KW, STRIDE, DILATION, padding)

    PADL, PADT = ci.PADL, ci.PADT

    OC, OH, OW = IC, ci.OH, ci.OW    

    O_IK_idxs = { idx:[ [ [], [] ], [ [], [] ] ] for idx in range(OH*OW) }
    K_IO_idxs = { idx:[ [ [], [] ], [ [], [] ] ] for idx in range(KH*KW) }
    I_KO_idxs = { idx:[ [ [], [] ], [ [], [] ] ] for idx in range(IH*IW) }

    for ow in range(OW):
      for oh in range(OH):
        O_idx = oh*OW + ow
        for kh in range(KH):
            for kw in range(KW):
              iw = -PADL + kw*DILATION + ow*STRIDE
              ih = -PADT + kh*DILATION + oh*STRIDE
              if (iw >= 0) & (ih >= 0) & (iw < IW) & (ih < IH):

                  O_IK_idxs[O_idx][0][0].append (ih)
                  O_IK_idxs[O_idx][0][1].append (iw)
                  O_IK_idxs[O_idx][1][0].append (kh)
                  O_IK_idxs[O_idx][1][1].append (kw)

                  K_idx = kh*KW + kw
                  K_IO_idxs[K_idx][0][0].append (ih)
                  K_IO_idxs[K_idx][0][1].append (iw)
                  K_IO_idxs[K_idx][1][0].append (oh)
                  K_IO_idxs[K_idx][1][1].append (ow)

                  I_idx = ih*IW + iw
                  I_KO_idxs[I_idx][0][0].append (kh)
                  I_KO_idxs[I_idx][0][1].append (kw)
                  I_KO_idxs[I_idx][1][0].append (oh)
                  I_KO_idxs[I_idx][1][1].append (ow)

    output_shape = (N, OC, OH, OW)
    output = np.empty( output_shape, np.float32)

    for n in range(N):
        for oc in range(OC):
            for oh in range(OH):
                for ow in range(OW):
                    O_idx = oh*OW + ow
                    I_idxs = O_IK_idxs[O_idx][0]
                    K_idxs = O_IK_idxs[O_idx][1]

                    v = ( input_n[ n,oc][..., I_idxs[0], I_idxs[1]] *
                        kernel_n [oc][..., K_idxs[0], K_idxs[1]] ).sum()

                    output[n,oc,oh,ow] = v

    dK = np.zeros(kernel_n.shape, np.float32)
    for ic in range(IC):
        for kh in range(KH):
            for kw in range(KW):
                K_idx = kh*KW + kw

                I_idxs = K_IO_idxs[K_idx][0]
                O_idxs = K_IO_idxs[K_idx][1]

                n_range = [*range(N)]
                v = (      input_n[n_range][...,ic, I_idxs[0], I_idxs[1]] *
                    conved_n_grad[n_range][...,ic, O_idxs[0], O_idxs[1]] ).sum()

                        
                dK[ic,kh,kw] = v

    dI = np.zeros(input_n.shape, np.float32)

    for n in range(N):
        for ic in range(IC):
            for ih in range(IH):
                for iw in range(IW):
                    I_idx = ih*IW + iw

                    K_idxs = I_KO_idxs[I_idx][0]
                    O_idxs = I_KO_idxs[I_idx][1]

                    v = (     kernel_n[ ic, K_idxs[0], K_idxs[1]] *
                         conved_n_grad[ n, ic, O_idxs[0], O_idxs[1]] ).sum()

                    dI[n,ic,ih,iw] = v

    return output, dI, dK
