import traceback
import math
import numpy as np
import litenn as nn
import litenn.core as nc
from litenn.core import CLKernelHelper as ph

def matmul_op(a_t, b_t, output_t=None, is_add_to_output=False): 
    """
    matmul operator in row-major format
    
     A(...,M,K) x B(...,K,N) = (...,M,N)  
     
     arguments
        output_t            compute result to this Tensor.
                            Tensor may be with different shape,
                            but should match total size.
                            gradfn will not be set.

        is_add_to_output    add result to output_t if output_t is set.
    """
    
    return matmulc_op(b_t, a_t, output_t=output_t, is_add_to_output=is_add_to_output)
def matmul(a_t, b_t):
    """
    matmul operator in row-major format

     A(...,M,K) x B(...,K,N) = (...,M,N)     
    """
    return matmulc_op(b_t, a_t)


def matmulc_op(a_t, b_t, output_t=None, is_add_to_output=False):
    """
    matmul operator in col-major format

        A(...,K,M) x B(...,N,K) = (...,N,M)

    arguments

        output_t            compute result to this Tensor.
                            Tensor may be with different shape,
                            but should match total size.
                            gradfn will not be set.

        is_add_to_output    add result to output_t if output_t is set.
    """
    is_add_to_output = False if output_t is None else is_add_to_output

    op = nc.Cacheton.get(_MatmulOp, a_t.shape, b_t.shape, is_add_to_output)

    if output_t is None:
        output_t = nn.Tensor ( op.output_shape )
        output_t._set_op_name ('matmul')
        output_t._assign_gradfn (a_t, lambda O_t, dO_t: matmul_a_grad(a_t, b_t, dO_t))
        output_t._assign_gradfn (b_t, lambda O_t, dO_t: matmul_b_grad(a_t, b_t, dO_t))
    elif output_t.shape.size != op.output_shape.size:
        raise ValueError(f'output_t must have size {op.output_shape.size}')

    op.forward_krn.run(a_t, b_t, output_t)

    return output_t
    
def matmul_a_grad(a_t, b_t, dO_t):
    b_t_axes = b_t.shape.axes_arange()    
    b_t = b_t.transpose( b_t_axes[:-2] + b_t_axes[-1:] + b_t_axes[-2:-1] )
    matmulc_op (dO_t, b_t, output_t=a_t.get_grad(), is_add_to_output=True)
    
def matmul_b_grad(a_t, b_t, dO_t):
    a_t_axes = a_t.shape.axes_arange()    
    a_t = a_t.transpose( a_t_axes[:-2] + a_t_axes[-1:] + a_t_axes[-2:-1] )
    matmulc_op (a_t, dO_t, output_t=b_t.get_grad(), is_add_to_output=True)
    
    

class _MatmulOp:
    def __init__(self, a_shape, b_shape, is_add_to_output):
        if a_shape.rank != b_shape.rank:
            raise ValueError(f'Ranks are not equal. {a_shape.rank} != {b_shape.rank}')
        
        rank = a_shape.rank
        if rank < 2:
            raise ValueError('Tensors rank must be at least 2.')
        
        K, M = a_shape[-2], a_shape[-1]
        N, B_COLS = b_shape[-2], b_shape[-1]

        if K != B_COLS:
            raise ValueError('A_ROWS != B_COLS')
        
        BATCH = a_shape[0:-2].size
        B_BATCH = b_shape[0:-2].size
        
        if BATCH != B_BATCH:
            raise ValueError(f'BATCH size {BATCH} != {B_BATCH} in shapes {a_shape} {b_shape}')
        
        if rank == 2:
            self.output_shape = output_shape = nc.TensorShape( (N, M) )
        else:
            self.output_shape = output_shape = nc.TensorShape( a_shape[:-2]+(N, M) )

        self.M = M
        self.N = N
        self.K = K

        # Determining optimal tile widths
        for MW in [16,8,4,2,1]:
            if M % MW == 0:
                break
        for KW in [8,4,2,1]:
            if N % KW == 0 and K % KW == 0:
                break
        NW = KW

        self.forward_krn = nc.CLKernel(global_shape=(M//MW, N//NW, BATCH), kernel_text=f"""
#define K {K}
#define N {N}
#define MW {MW}     // M tile Width
#define NW {NW}     // N tile Width  -- NW & KW should be the same !
#define KW {KW}     // K tile Width
#define MT {M//MW}  // MT is max for 'mt' (M tile count)
#define KT {K//KW}  // KT is max for 'kt' (K tile count)

#define floatMW { f'float{MW}' if MW != 1 else 'float'}
#define floatKW { f'float{KW}' if KW != 1 else 'float'}

__kernel void GeMM(const __global floatMW* restrict A, const __global floatKW* restrict B, __global floatMW* C)
{{
    size_t mt = get_global_id(0);    //global M-tile id
    size_t nc = get_global_id(1);    //global N-tile id
    size_t batch = get_global_id(2); 
        
    float AT[KW][MW]; // sub tiles
    float BT[NW][KW];
    float CT[NW][MW];

    #pragma unroll
    for (uint i=0; i<NW*MW; ++i) // zero CT tile
        ((float*) CT)[i] = 0.0;

    for (uint kt=0; kt<KT; ++kt)  // iterate over K-dim tiles
    {{
        #pragma unroll
        for (uint k=0; k<KW; ++k)  // every k-element inside K-dim tile
            *( (floatMW*) AT[k] ) = A[batch*K*MT + (kt*KW + k)*MT + mt]; // store M-Width floats

        #pragma unroll
        for (uint n=0; n<NW; ++n)  // every n-element inside N-dim tile
            *( (floatKW*) BT[n] ) = B[batch*N*KT + (nc*NW + n)*KT + kt]; // store K-Width floats

        #pragma unroll
        for (uint k=0; k<KW; ++k)
        #pragma unroll
        for (uint n=0; n<NW; ++n)  // sub tiles multiplication
        #pragma unroll
        for (uint m=0; m<MW; ++m)
            CT[n][m] += AT[k][m] * BT[n][k];
    }}

    #pragma unroll
    for (uint n=0; n<NW; ++n)
        C[ batch*N*MT + (nc*NW + n)*MT + mt] {'+=' if is_add_to_output else '='}
                               *( (floatMW*) CT[n]);
}}""")


def matmul_test():
    for _ in range(10):
        try:
            BATCH = np.random.randint(8)+1
            M = np.random.randint(8)+1
            N = np.random.randint(32768)+1
            K = np.random.randint(32768)+1

            while K*N > ( 8000000 // BATCH ):
                K = max(1, K // 2)
                N = max(1, N // 2)

            if np.random.randint(2) == 0:
                size = [2,4,8,16][np.random.randint(4)]
                M = max(1, M // size) * size
                N = max(1, N // size) * size
                K = max(1, K // size) * size
                
            if BATCH == 1:
                A_shape = (M, K)
                B_shape = (K, N)
            else:
                A_shape = (BATCH, M, K)
                B_shape = (BATCH, K, N)
                
            A_n = np.random.randint( 2**4, size=A_shape ).astype(np.float32)
            B_n = np.random.randint( 2**4, size=B_shape ).astype(np.float32)

            O_n = np.matmul(A_n, B_n)

            A_t = nn.Tensor_from_value(A_n)
            B_t = nn.Tensor_from_value(B_n)
            O_t = nn.matmul(A_t, B_t)
            if O_n.shape != O_t.shape:
                raise Exception('shape is not equal')            
            if not all ( np.ndarray.flatten( O_t.np() == O_n) ):
                raise Exception(f'data is not equal')

            O_n_grad = np.random.randint( 2**3, size=O_n.shape).astype(np.float32)
            
            b_n_axes = tuple(np.arange( len(B_n.shape) ))
            B_n_T = B_n.transpose( b_n_axes[:-2] + b_n_axes[-1:] + b_n_axes[-2:-1] )
            
            a_n_axes = tuple(np.arange( len(A_n.shape) ))
            A_n_T = A_n.transpose( a_n_axes[:-2] + a_n_axes[-1:] + a_n_axes[-2:-1] )
            
            A_n_grad = np.matmul( O_n_grad, B_n_T)
            B_n_grad = np.matmul( A_n_T, O_n_grad)

            A_t.get_grad().fill(1.0)
            B_t.get_grad().fill(1.0)

            nn.backward( { O_t : O_n_grad}, grad_for_non_trainables=True )

            if not all ( np.ndarray.flatten( (A_t.get_grad().np()-1.0) == A_n_grad) ):
                raise Exception(f'dA is not equal')

            if not all ( np.ndarray.flatten( (B_t.get_grad().np()-1.0) == B_n_grad) ):
                raise Exception(f'dB is not equal')
        except:
            raise Exception(f"""
M  : {M}
N  : {N}
K  : {K}
O_n.shape  : {O_n.shape}
O_t.shape  : {O_t.shape}
{traceback.format_exc()}
""")