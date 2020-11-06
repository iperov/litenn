import traceback
import math
import numpy as np

import litenn as nn
import litenn.core as nc
from litenn.core import CLKernelHelper as ph

def avg_pool2D (input_t, pool_size=2, stride=2, padding='same'): 
    """
    Average pooling 2D operator.
    
    arguments
    
        pool_size(2)    int
        
        stride(2)       int
        
        padding(same)   'valid'         no padding
                        'same'          output size will be the same 
                                        or divided by stride
                                        
                        int             padding value for all sides
                        tuple of 4 ints paddings for left,top,right,bottom sides        
    """
    return pool2D_op ('avg', input_t, pool_size, stride, padding)
    
def min_pool2D (input_t, pool_size=2, stride=2, padding='same'): 
    """
    Min pooling 2D operator.
    
    arguments
    
        pool_size(2)    int
        
        stride(2)       int
        
        padding(same)   'valid'         no padding
                        'same'          output size will be the same 
                                        or divided by stride
                                        
                        int             padding value for all sides
                        tuple of 4 ints paddings for left,top,right,bottom sides        
    """
    return pool2D_op ('min', input_t, pool_size, stride, padding)
    
def max_pool2D (input_t, pool_size=2, stride=2, padding='same'): 
    """
    Max pooling 2D operator.
    
    arguments
    
        pool_size(2)    int
        
        stride(2)       int
        
        padding(same)   'valid'         no padding
                        'same'          output size will be the same 
                                        or divided by stride
                                        
                        int             padding value for all sides
                        tuple of 4 ints paddings for left,top,right,bottom sides        
    """
    return pool2D_op ('max', input_t, pool_size, stride, padding)

def pool2D_op (op_type, input_t, pool_size=2, stride=2, padding='same'):
    """
    arguments
    
        op_type     'avg','min','max'
    """
    op = nc.Cacheton.get(_Pool2DOp, op_type, input_t.shape, int(pool_size), int(stride), padding)

    output_t = nn.Tensor( op.output_shape )
    output_t._set_op_name(f'{op_type}_pool2D')
    output_t._assign_gradfn (input_t, lambda O_t, dO_t: pool2D_dI_gradfn(op, input_t, dO_t, O_t) )

    op.O_forward_krn.run(output_t, input_t)
    return output_t

def pool2D_dI_gradfn(op, input, dO, O_t):
    if op.op_type == 'avg':
        op.dI_krn.run(input.get_grad(), dO)
    elif op.op_type in ['min', 'max']:
        op.dI_krn.run(input.get_grad(), input, dO, O_t)

class _Pool2DOp:
    def __init__(self, op_type, input_shape : nc.TensorShape, pool_size, stride, padding):
        if op_type not in ['avg','min','max']:
            raise ValueError (f'unknown op_type {op_type}')
        if pool_size < 2:
            raise ValueError(f'pool_size {pool_size} must be at least 2')
        self.op_type = op_type
        
        N,IC,IH,IW = input_shape 
        ci = nc.info.InfoConv2D(IH,IW, pool_size, pool_size, stride, 1, padding)
        OC, OH, OW = IC, ci.OH, ci.OW
        self.output_shape = output_shape = nc.TensorShape( (N, OC, OH, OW) )

        common_kernel_text = f"""
{ph.define_axes_accessor('I', input_shape, 'NCHW')}
{ph.define_axes_accessor('O', output_shape, 'NCHW')}

#define PADL {ci.PADL}
#define PADT {ci.PADT}

#define POOL_SIZE {pool_size}
#define STRIDE {stride}
"""

        self.O_forward_krn = nc.CLKernel(global_shape=(output_shape.size,), kernel_text=f"""
{common_kernel_text}

__kernel void impl(__global float* O, __global const float* I)
{{
    size_t gid = get_global_id(0);
    {ph.axes_idxs_from_var('o', 'NCHW', 'gid')}

    { {'avg' : 'float v = 0.0; int v_count = 0;',
       'max' : 'float v = -INFINITY;',
       'min' : 'float v = INFINITY;',
      }[op_type] }

    for (int ph=0; ph<POOL_SIZE; ++ph)
    for (int pw=0; pw<POOL_SIZE; ++pw)
    {{
        int ih = -PADT + ph + oh*STRIDE;
        int iw = -PADL + pw + ow*STRIDE;
        if (iw >= 0 & ih >= 0 & iw < IW & ih < IH)
        {{
            { {'avg' : 'v +=        I[I_idx(on,oc,ih,iw)]; ++v_count;',
               'max' : 'v = fmax(v, I[I_idx(on,oc,ih,iw)]);',
               'min' : 'v = fmin(v, I[I_idx(on,oc,ih,iw)]);',
              }[op_type] }
        }}
    }}

    { {'avg' : 'if (v_count != 0) v /= v_count;',
       'max' : 'if (v == -INFINITY) v = 0.0;',
       'min' : 'if (v == INFINITY) v = 0.0;',
      }[op_type] }


    O[gid] = v;
}}
""")
        if op_type == 'avg':
            self.dI_krn = nc.CLKernel(global_shape=(input_shape.size,), kernel_text=f"""
{common_kernel_text}
__kernel void impl(__global float* dI, __global const float* dO)
{{
    size_t gid = get_global_id(0);
    {ph.axes_idxs_from_var('i', 'NCHW', 'gid')}

    float v = 0.0;

    for (int ph=0; ph<POOL_SIZE; ++ph)
    for (int pw=0; pw<POOL_SIZE; ++pw)
    {{
        int oh = (PADT + ih - ph ) / STRIDE;
        int ow = (PADL + iw - pw ) / STRIDE;
        if (ow >= 0 & oh >= 0 & ow < OW & oh < OH
            & iw == (-PADL + pw + ow*STRIDE)
            & ih == (-PADT + ph + oh*STRIDE) )
        {{
            int d=0;
            for (int dph=0; dph<POOL_SIZE; ++dph)
            for (int dpw=0; dpw<POOL_SIZE; ++dpw)
            {{
                int dih = -PADT + dph + oh*STRIDE;
                int diw = -PADL + dpw + ow*STRIDE;
                d += (diw >= 0 & dih >= 0 & diw < IW & dih < IH);
            }}
            v += dO[O_idx(in,ic,oh,ow)] / d;
        }}
    }}

    dI[gid] += v;
}}
""")
        elif op_type in ['min','max']:
            # Implementation is different from tensorflow in case when the same values exist in reduction axes.
            # Example tf     : 3 4 5 5 , max = 5, gradients : 0 0 0.5 0.5
            # Example litenn : 3 4 5 5 , max = 5, gradients : 0 0   1   0
            #                                 or  gradients : 0 0   0   1 - depends on which GPU thread will be first !
            self.dI_krn = nc.CLKernel(global_shape=(input_shape.size,), kernel_text=f"""
{common_kernel_text}
__kernel void impl(__global float* dI, __global const float* I, __global const float* dO, __global const float* O)
{{
    size_t gid = get_global_id(0);
    {ph.axes_idxs_from_var('i', 'NCHW', 'gid')}

    float v = 0.0;

    // Iterate over all O pixels, where 'I' have contribution
    for (int ph=0; ph<POOL_SIZE; ++ph)
    for (int pw=0; pw<POOL_SIZE; ++pw)
    {{
        int oh = (PADT + ih - ph ) / STRIDE;
        int ow = (PADL + iw - pw ) / STRIDE;
        if (ow >= 0 & oh >= 0 & ow < OW & oh < OH
            & iw == (-PADL + pw + ow*STRIDE)
            & ih == (-PADT + ph + oh*STRIDE) )
        {{
            //Now we have oh,ow where ih,iw have contribution

            float Ov = O[O_idx(in,ic,oh,ow)];

            //Iterate in those I pixels, which were used to produce O
            //to determine first min/max match
            for (int dphw=0; dphw < POOL_SIZE*POOL_SIZE; ++dphw)
            {{
                int dih = -PADT + (dphw / POOL_SIZE) + oh*STRIDE;
                int diw = -PADL + (dphw % POOL_SIZE) + ow*STRIDE;
                if (diw >= 0 & dih >= 0 & diw < IW & dih < IH &
                    I[I_idx(in,ic,dih,diw)] == Ov)
                {{
                    // Match I==O
                    if (dih == ih & diw == iw)
                        // but add gradient only if current ih/iw index match dih/diw
                        v += dO[O_idx(in,ic,oh,ow)];
                    break;
                }}
            }}
        }}
    }}

    dI[gid] += v;
}}
""")

def pool2d_test():
    for batch in [1,4]:
        for in_ch in [1,2,4]:
            for w,h in zip(*[[4,8,16]]*2):
              for pool_size in [2,3]:
                for stride in [1,2,3]:
                    for padding in ['same','valid',0,1,2]:
                      for op_type in ['avg','max','min']:
                        try:
                            input_shape  = (batch, in_ch, h, w)
                            if op_type == 'avg':
                                input_n  = np.random.randint( 2**4, size=input_shape ).astype(np.float32)
                            else:
                                # for minmax make unique values in order not to test 'same-hit'
                                input_shape_prod = int(np.prod(input_shape))
                                input_n = np.arange(input_shape_prod, dtype=np.float32).reshape(input_shape) / input_shape_prod

                            input_t  = nn.Tensor_from_value(input_n)

                            pooled_n, input_n_grad = _numpy_pool2d(op_type, input_n, pool_size, STRIDE=stride, padding=padding)

                            if op_type == 'avg':
                                pooled_t = nn.avg_pool2D (input_t, pool_size, stride, padding)
                            elif op_type == 'max':
                                pooled_t = nn.max_pool2D (input_t, pool_size, stride, padding)
                            elif op_type == 'min':
                                pooled_t = nn.min_pool2D (input_t, pool_size, stride, padding)

                            if pooled_n.shape != pooled_t.shape:
                                raise Exception(f'shape is not equal')

                            if np.sum(np.abs (pooled_t.np() - pooled_n)) > 0.1:
                                raise Exception(f'data is not equal')

                            input_t.get_grad().fill(1.0)
                            nn.backward(pooled_t, grad_for_non_trainables=True )

                            if sum (np.ndarray.flatten( input_t.get_grad().np()-1.0 - input_n_grad )) >= 1.0:
                                raise Exception('grad is not equal')

                        except:
                            raise Exception(f"""
op_type        : {op_type}
input_shape    : {input_shape}
pool_size      : {pool_size}
stride         : {stride}
padding        : {padding}
pooled_n.shape : {pooled_n.shape}
pooled_t.shape : {pooled_t.shape}
{traceback.format_exc()}
""")



def _numpy_pool2d(op_type, input_n, POOL_SIZE, STRIDE=1, padding='same'):
    N,IC,IH,IW = input_n.shape 
    ci = nc.info.InfoConv2D(IH,IW, POOL_SIZE, POOL_SIZE, STRIDE, 1, padding)
    OC, OH, OW = IC, ci.OH, ci.OW    
    PADL, PADT = ci.PADL, ci.PADT

    O_I_idxs = { idx:[] for idx in range(OH*OW) }
    I_O_idxs = { idx:[] for idx in range(IH*IW) }

    for ow in range(OW):
      for oh in range(OH):
        O_idx = oh*OW + ow
        for ph in range(POOL_SIZE):
            for pw in range(POOL_SIZE):
              iw = -PADL + pw + ow*STRIDE
              ih = -PADT + ph + oh*STRIDE
              if (iw >= 0) & (ih >= 0) & (iw < IW) & (ih < IH):
                  I_idx = ih*IW + iw
                  O_I_idxs[O_idx].append ( (ih, iw) )
                  I_O_idxs[I_idx].append ( (oh, ow) )

    output_shape = (N, OC, OH, OW)
    output = np.empty( output_shape, np.float32)

    input_n_grad = np.zeros (input_n.shape, dtype=np.float32)
    for n in range(N):
        for oc in range(OC):
            for oh in range(OH):
                for ow in range(OW):
                    O_idx = oh*OW + ow
                    I_idxs = O_I_idxs[O_idx]

                    v = 0.0
                    if len(I_idxs) != 0:
                        if op_type == 'max':
                            v = float("-inf")
                        elif op_type == 'min':
                            v = float("inf")

                        minmax_idx = -1
                        for idx, (ih,iw) in enumerate(I_idxs):
                            iv = input_n[n,oc, ih, iw]
                            if op_type == 'avg':
                                v += iv
                            elif op_type == 'max':
                                if iv > v:
                                    v = iv
                                    minmax_idx = idx
                            elif op_type == 'min':
                                if iv < v:
                                    v = iv
                                    minmax_idx = idx

                        if op_type == 'avg':
                            for ih,iw in I_idxs:
                                input_n_grad[n,oc, ih, iw] += 1.0 / len(I_idxs)
                        elif op_type == 'max' or op_type == 'min':
                            ih,iw = I_idxs[minmax_idx]
                            input_n_grad[n,oc, ih, iw] += 1.0

                        if op_type == 'avg':
                            v /= len(I_idxs)
                        elif op_type == 'max':
                            if v == float("-inf"):
                                v = 0.0
                        elif op_type == 'min':
                            if v == float("inf"):
                                v = 0.0
                    output[n,oc,oh,ow] = v

    return output, input_n_grad