import litenn as nn
import litenn.core as nc
from litenn.core import CLKernelHelper as ph


def unfold2D(input_t, N, C, H, W, OH, OW, KH, KW, PADL, PADT, DILATION, STRIDE, sshape, is_transpose, input_data_format='NCHW', output_t=None, is_add_to_output=False):
    """
    Unfolds 2D image to desired sshape.

    arguments
    
        N, C, H, W, OH, OW, KH, KW, PADL, PADT, DILATION, STRIDE
                conv int parameters
        
        sshape          string containing symbols:
                            NCHWJI_
                            
                            where 
                            N : batch
                            C : channels
                            HW : height/width
                            JI : kernel height/width
                            _  : axis separator
         
        is_transpose        True if transpose operation.
                           
    example of sshape:
    
     'CJI_NHW'    = (C*J*I,N*H*W)
     'NHW_CJI'    = (N*H*W,C*J*I)
     'N_C_H_W_JI' = (N,C,H,W,J*I)     
     
    This operator does not have gradient and should not be used in backprop directly.
    """
    is_add_to_output = False if output_t is None else is_add_to_output
    
    N, C, H, W, OH, OW, KH, KW, PADL, PADT, DILATION, STRIDE = (int(x) for x in (N, C, H, W, OH, OW, KH, KW, PADL, PADT, DILATION, STRIDE))
    
    op = nc.Cacheton.get(_Unfold2DOp, N, C, H, W, OH, OW, KH, KW, PADL, PADT, DILATION, STRIDE, sshape, is_transpose, input_data_format, is_add_to_output)

    if output_t is None:
        output_t = nn.Tensor(op.output_shape)
        output_t._set_op_name('unfold2D')
    elif output_t.shape.size != op.output_shape.size:
        raise ValueError(f'output_t must have size {op.output_shape.size}')

    op.krn.run (output_t, input_t)
    return output_t
        
class _Unfold2DOp:

    def __init__(self, N, C, H, W, OH, OW, KH, KW, PADL, PADT, DILATION, STRIDE, sshape, is_transpose, input_data_format, is_add_to_output):
        if input_data_format not in ['NCHW', 'NHWC']:
            raise ValueError(f'Unknown input_data_format {input_data_format}.')
            
        d = {'N' : N, 'C' : C, 'H' : OH, 'W' : OW, 'J' : KH, 'I' : KW}        
        sshape = sshape.upper()
        
        output_shape = [1]
        O_shape = []
        O_shape_size = 1
        O_shape_axes = ''
        for symbol in sshape:
            if symbol == '_':
                output_shape.append(1)
            else:
                value = d.get(symbol, None)                
                if value is None:
                    raise ValueError(f'Unknown symbol {symbol}. Valid symbols: _{list(d.keys())}')
                if value == -1:
                    raise ValueError(f'Duplicate symbol {symbol}')                
                output_shape[-1] *= value
                O_shape.append(value)
                O_shape_size *= value
                O_shape_axes += symbol
                d[symbol] = -1
        
        self.output_shape = tuple(output_shape)
        O_shape = tuple(O_shape)

        input_shape = (N,C,H,W) if input_data_format == 'NCHW' else (N,H,W,C)
        
        
        if not is_transpose:
            self.krn = nc.CLKernel(global_shape=(O_shape_size,), kernel_text=f"""
#define STRIDE {STRIDE}
#define DILATION {DILATION}
#define PADT {PADT}
#define PADL {PADL}

{ph.define_axes_accessor('I', input_shape, input_data_format)}
{ph.define_axes_accessor('O', O_shape, O_shape_axes)}
__kernel void impl(__global float* O, __global const float* I)
{{  
    size_t gid = get_global_id(0);
    {ph.axes_idxs_from_var('o', O_shape_axes, 'gid')}
#define ic oc
#define in on
    float value = 0.0;
    int ih = -PADT + oj*DILATION + oh*STRIDE;
    if (ih >= 0 & ih < IH)
    {{
        int iw = -PADL + oi*DILATION + ow*STRIDE;
        if (iw >= 0 & iw < IW)
            value = I[I_idx({ph.axes_order_enum('i', input_data_format)})];
    }}

    O[O_idx({ph.axes_order_enum('o', O_shape_axes)})]
    {'+=' if is_add_to_output else '='} value;
}}
""")
        else:
            self.krn = nc.CLKernel(global_shape=(O_shape_size,), kernel_text=f"""
#define STRIDE {STRIDE}
#define DILATION {DILATION}
#define PADT {PADT}
#define PADL {PADL}

{ph.define_axes_accessor('I', input_shape, input_data_format)}
{ph.define_axes_accessor('O', O_shape, O_shape_axes)}
__kernel void impl(__global float* O, __global const float* I)
{{  
    size_t gid = get_global_id(0);
    {ph.axes_idxs_from_var('o', O_shape_axes, 'gid')}
#define ic oc
#define in on
    float value = 0.0;
    int ih = ( PADT + oh - oj*DILATION ) / STRIDE;
    if (ih >= 0 & ih < IH & (oh == -PADT + oj*DILATION + ih*STRIDE) )
    {{
        int iw = ( PADL + ow - oi*DILATION ) / STRIDE;   
        if (iw >= 0 & iw < IW & (ow == -PADL + oi*DILATION + iw*STRIDE) )
            value = I[I_idx(in,ic,ih,iw)];
    }}
    O[O_idx({ph.axes_order_enum('o', O_shape_axes)})]
    {'+=' if is_add_to_output else '='} value;
}}
""")
