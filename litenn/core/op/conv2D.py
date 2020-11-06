import traceback
import math
import numpy as np

import litenn as nn
import litenn.core as nc
from litenn.core import CLKernelHelper as ph

def conv2D (input_t, kernel_t, stride=1, dilation=1, padding='same'):
    """
    Conv2D operator.

     input_t     Tensor          shape must be 
                                 (batch, in_ch,  height,  width)
     
     kernel_t    Tensor          shape must be 
                                 (out_ch,in_ch,k_height,k_width)
 
     stride(1)       int
     
     dilation(1)     int
     
     padding(same)   'valid'         no padding
                     'same'          output size will be the same 
                                     or divided by stride
                     int             padding value for all sides
                     Iterable of 4 ints 
                                paddings for left,top,right,bottom sides

    """
    
    op = nc.Cacheton.get(_Conv2DOp, input_t.shape, kernel_t.shape, int(stride), int(dilation), padding)

    output_t = nn.Tensor( op.output_shape )
    output_t._set_op_name('conv2D')
    output_t._assign_gradfn (input_t,  lambda O_t, dO_t: conv2D_dI_gradfn(op, input_t, kernel_t, dO_t) )
    output_t._assign_gradfn (kernel_t, lambda O_t, dO_t: conv2D_dK_gradfn(op, input_t, kernel_t, dO_t) )

    out = nc.op.matmul( kernel_t.reshape( (kernel_t.shape[0], -1) ),
                        op.im2col(input_t) )

    nc.op.transpose( out.reshape(op.OC_N_OH_OW), (1,0,2,3), output_t=output_t)

    return output_t

def conv2D_dI_gradfn(op, input_t, kernel_t, dO_t):        
    kernel_t = kernel_t.transpose((0,2,3,1))
    kernel_t = kernel_t.reshape( (-1,kernel_t.shape[-1]) ) 
    dI = nc.op.matmul(op.im2rowT(dO_t), kernel_t).reshape(op.N_IH_IW_IC)
    nc.op.transpose( dI, (0,3,1,2), output_t=input_t.get_grad(), is_add_to_output=True)

def conv2D_dK_gradfn(op, input_t, kernel_t, dO_t):
    nc.op.matmul ( dO_t.transpose((1,0,2,3)).reshape(op.OC_NxOHxOW),
                   op.im2row(input_t), output_t=kernel_t.get_grad(), is_add_to_output=True)
        
class _Conv2DOp():
    def __init__(self, input_shape, kernel_shape, stride, dilation, padding):
        N,IC,IH,IW = input_shape
        KO,KI,KH,KW = kernel_shape
        if KI != IC:
            raise ValueError(f'Kernel input channels {KI} does not match tensor input channels {IC}.')
        
        ci = nc.info.InfoConv2D(IH, IW, KH, KW, stride, dilation, padding)
        OC, OH, OW = KO, ci.OH, ci.OW
        self.output_shape = output_shape = nc.TensorShape( (N, OC, OH, OW) )

        self.OC_N_OH_OW = (OC,N,OH,OW)
        self.OC_NxOHxOW = (OC,N*OH*OW)
        self.N_IH_IW_IC = (N,IH,IW,IC)

        self.im2col  = lambda x: nc.op.unfold2D (x, N, IC, IH, IW, OH, OW, KH, KW, ci.PADL, ci.PADT, dilation, stride, 'CJI_NHW', is_transpose=False)
        self.im2row  = lambda x: nc.op.unfold2D (x, N, IC, IH, IW, OH, OW, KH, KW, ci.PADL, ci.PADT, dilation, stride, 'NHW_CJI', is_transpose=False)       
        self.im2rowT = lambda x: nc.op.unfold2D (x, N, OC, OH, OW, IH, IW, KH, KW, ci.PADL, ci.PADT, dilation, stride, 'NHW_CJI', is_transpose=True)  


def conv2d_test():
    for padding in ['same','valid',0,1,2]:
      for dilation in [1,2]:
        for stride in [1,2,3]:
          for ks in [7,5,3,1]:
            for n in [1,4]:
              for ic in [1,2,4]:
                for oc in [1,2,4]:
                  for ih,iw in zip(*[[4,8,16]]*2):                      
                    if padding == 'valid' and iw < ks:
                        continue
                    try:
                        input_shape  = (n, ic, ih, iw)
                        kernel_shape = (oc, ic, ks, ks)

                        input_n  = np.random.randint( 2**4, size=input_shape ).astype(np.float32)
                        kernel_n = np.random.randint( 2**4, size=kernel_shape ).astype(np.float32)

                        input_t  = nn.Tensor_from_value(input_n)
                        kernel_t = nn.Tensor_from_value(kernel_n)

                        conved_t = nn.conv2D(input_t, kernel_t, stride=stride, dilation=dilation, padding=padding)
                        conved_n_grad = np.random.randint( 2**4, size=conved_t.shape).astype(np.float32)
                        conved_n, dI_val, dK_val = _numpy_conv2d(input_n, kernel_n, conved_n_grad, STRIDE=stride, DILATION=dilation, padding=padding)

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

def _numpy_conv2d(input_n, kernel_n, conved_n_grad, STRIDE=1, DILATION=1, padding='same'):
    N, IC, IH, IW = input_n.shape
    KO, KI, KH, KW = kernel_n.shape
    
    ci = nc.info.InfoConv2D(IH, IW, KH, KW, STRIDE, DILATION, padding)

    PADL, PADT = ci.PADL, ci.PADT

    OC, OH, OW = KO, ci.OH, ci.OW    

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
                    
                    ic_range = [*range(IC)]
                    v = ( input_n[ n,ic_range][..., I_idxs[0], I_idxs[1]] *
                         kernel_n[oc,ic_range][..., K_idxs[0], K_idxs[1]] ).sum()

                    output[n,oc,oh,ow] = v

    dK = np.zeros(kernel_n.shape, np.float32)
    for oc in range(OC):
        for ic in range(IC):
            for kh in range(KH):
                for kw in range(KW):
                    K_idx = kh*KW + kw

                    I_idxs = K_IO_idxs[K_idx][0]
                    O_idxs = K_IO_idxs[K_idx][1]

                    n_range = [*range(N)]
                    v = (      input_n[n_range][...,ic, I_idxs[0], I_idxs[1]] *
                         conved_n_grad[n_range][...,oc, O_idxs[0], O_idxs[1]] ).sum()
                            
                    dK[oc,ic,kh,kw] = v

    dI = np.zeros(input_n.shape, np.float32)

    for n in range(N):
        for ic in range(IC):
            for ih in range(IH):
                for iw in range(IW):
                    I_idx = ih*IW + iw

                    K_idxs = I_KO_idxs[I_idx][0]
                    O_idxs = I_KO_idxs[I_idx][1]

                    oc_range = [*range(OC)]
                    v = (     kernel_n[ oc_range][..., ic, K_idxs[0], K_idxs[1]] *
                         conved_n_grad[ n, oc_range][..., O_idxs[0], O_idxs[1]] ).sum()

                    dI[n,ic,ih,iw] = v

    return output, dI, dK
