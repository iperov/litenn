
class CLKernelHelper:
    """
    Helper to format CL kernels.
    """
    
    @staticmethod
    def define_axes_accessor(axis_letter, shape, axes_symbols=None):
        """
        Returns a definitions of shape accesor
        
        arguments
            
            axis_letter     text symbol A-Z in any case.
            
            shape           Iterable
            
            axes_symbols(None)  string of symbols. 
                                None -> numeric symbols will be used
            
         example for 'i', TensorAxes((4,512))
         
            #define I0 4
            #define I1 512            
            #define I_idx(i0,i1) ((size_t)i0)*I1+i1
            //access by idx with modulus
            #define I_idx_mod(i0,i1) MODULO(((size_t)i0), I0)*I1 + MODULO(i1, I1)
        """
        shape = tuple(shape)
        rank = len(shape)
        
        if axes_symbols is None:
            axes_symbols = "".join([str(i) for i in range(rank)])
        axes_symbols = axes_symbols.upper()
        
        out = '#define MODULO(x,N) (x % N)\n'

        for i in range(rank):
            out += f'#define {axis_letter.upper()}{axes_symbols[i]} {shape[i]}\n'        
        
        out += f'#define {axis_letter.upper()}_idx({CLKernelHelper.axes_seq_enum(axis_letter, rank)}) '
        
        for i in range(rank):
            if i == 0:
                out += f'((size_t)({axis_letter.lower()}{i}))'
            else:
                out += f'({axis_letter.lower()}{i})'
            
            for j in range(i+1,rank):
                out += f'*{axis_letter.upper()}{axes_symbols[j]}'
            if i != rank-1:
                out += '+'
            
        out += '\n'
                
        out += f'#define {axis_letter.upper()}_idx_mod('
        out += CLKernelHelper.axes_seq_enum(axis_letter, rank)
        out += ") "
        
        for i in range(rank):
            if i == 0:
                out += f'MODULO( ((size_t)({axis_letter.lower()}{i})) ,{axis_letter.upper()}{axes_symbols[i]})'
            else:
                out += f'MODULO( ({axis_letter.lower()}{i}),{axis_letter.upper()}{axes_symbols[i]})'
            
            for j in range(i+1,rank):
                out += f'*{axis_letter.upper()}{axes_symbols[j]}'
            if i != rank-1:
                out += '+'
                
        out += "\n"
        return out
    
    @staticmethod    
    def define_axes_sizes(axis_letter, axes_sizes):
        """
        Returns a text of axes sizes, example
        #define I0 4
        #define I1 512
        #define I2 512
        """
        out = ""
        axes_sizes = tuple(axes_sizes)        
        ndim = len(axes_sizes)
        for i in range(ndim):
            out += f'#define {axis_letter.upper()}{i} {axes_sizes[i]}\n'        
        
        return out
   
    @staticmethod         
    def axes_idxs_from_var(axis_letter, rank_or_axes_symbols, var_name):
        """
        decompose a size_t variable to axes indexes.
        Keeps original variable untouched.
        
        Example
         'i',3,'gid'
         size_t gid_original = gid;
         size_t i2 = gid % I2; gid /= I2;
         size_t i1 = gid % I1; gid /= I1;
         size_t i0 = gid % I0; gid = gid_original;
         
         'i','HW','gid'
         size_t gid_original = gid;
         size_t iw = gid % IW; gid /= IW;
         size_t ih = gid % IH; gid = gid_original;
        """
        
        if isinstance(rank_or_axes_symbols, int):
            rank = rank_or_axes_symbols
            axes_symbols = "".join([str(i) for i in range(rank)])
        elif isinstance(rank_or_axes_symbols, str):
            rank = len(rank_or_axes_symbols)
            axes_symbols = rank_or_axes_symbols
        else:
            raise ValueError(f'Unknown type of rank_or_axes_symbols')
            
        out = f'size_t {var_name}_original = {var_name};'
        
        for i in range(rank-1,-1,-1):
            if i == 0:
                if rank > 1:
                    out += f'size_t {axis_letter.lower()}{axes_symbols[i].lower()} = {var_name} / {axis_letter.upper()}{axes_symbols[i+1].upper()};'
                else:
                    out += f'size_t {axis_letter.lower()}{axes_symbols[i].lower()} = {var_name};'
            else:
                out += f'size_t {axis_letter.lower()}{axes_symbols[i].lower()} = {var_name} % {axis_letter.upper()}{axes_symbols[i].upper()};'
                
            if i > 1:
                out += f' {var_name} /= {axis_letter.upper()}{axes_symbols[i].upper()};\n'
        out += f'{var_name} = {var_name}_original;\n'
        return out
    
    @staticmethod    
    def axes_order_enum(axis_letter, axes_order):
        """
        returns axis enumeration with given order
        
        Example        
         ('i', (1,2,0)) returns 'i1,i2,i0'         
         ('i', 'HW') return 'ih,iw'
        """
        if isinstance(axes_order, str):
            axes_order = axes_order.lower()
        else:
            axes_order = tuple(axes_order)
        
        return ','.join( [ f'{axis_letter.lower()}{axes_order[axis]}' for axis in range(len(axes_order)) ])
       
    @staticmethod 
    def axes_seq_enum(axis_letter, rank, new_axis=None, zero_axes=None):
        """
        returns axis sequental enumeration with given rank
        
        Example
        
         ('i', 4) returns 'i0,i1,i2,i3'
            
         ('i', 4, new_axis=('name',1) ) returns 'i0,name,i1,i2,i3'
         
         ('i', 3, zero_axes=(1,) ) returns 'i0,0,i2'
        """
        
        if zero_axes is not None:
            axes = [ '0' if axis in zero_axes else f'{axis_letter.lower()}{axis}' for axis in range(rank) ]
        else:
            axes = [ f'{axis_letter.lower()}{axis}' for axis in range(rank) ]
        
        if new_axis is not None:
            name, axis = new_axis            
            return','.join(axes[:axis] + [name] + axes [axis:])            
        else:
            return ','.join(axes)
        
    @staticmethod
    def include_constants_pi():
        """
        defines PI constants
        
         PI_F
         PI_2_F
         PI_4_F
        """
        return f"""
#define  PI_F          3.14159274101257f
#define  PI_2_F        1.57079637050629f
#define  PI_4_F        0.78539818525314f
"""
      
    @staticmethod
    def include_hash():
        """
        returns hash functions:
        
         uint  hash_uint_uint(uint v)
         uint2 hash_uint2_uint2(uint2 v)
         uint3 hash_uint3_uint3(uint3 v) 
         
         float hash_float_uint(uint q) 
         float2 hash_float2_uint(uint q)
         float3 hash_float3_uint (uint v)
         
         float hash_float_float(float p)
         float hash_float_float2(float2 p)
        """

        return f"""
//---------- PCG hashes from https://www.shadertoy.com/view/XlGcRh

#define UIF (1.0 / (float)(0xffffffffU))
uint hash_uint_uint(uint v)
{{
    uint state = v * 747796405u + 2891336453u;
    uint word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return (word >> 22u) ^ word;
}}

uint2 hash_uint2_uint2 (uint2 v)
{{
    v = v * 1664525u + 1013904223u;
    v.x += v.y * 1664525u;
    v.y += v.x * 1664525u;
    v ^= v>>16u;
    v.x += v.y * 1664525u;
    v.y += v.x * 1664525u;
    v ^= v>>16u;
    return v;
}}

uint3 hash_uint3_uint3(uint3 v) 
{{
    v = v * 1664525u + 1013904223u;
    v.x += v.y*v.z;
    v.y += v.z*v.x;
    v.z += v.x*v.y;
    v ^= v >> 16u;
    v.x += v.y*v.z;
    v.y += v.z*v.x;
    v.z += v.x*v.y;
    return v;
}}

float hash_float_uint(uint v)
{{   
	return (float)( hash_uint_uint(v) ) * UIF;
}}

float2 hash_float2_uint (uint v)
{{
    uint2 q = hash_uint2_uint2( (uint2)(v, 1) );    
    return (float2)(q.x, q.y) * UIF;
}}

float3 hash_float3_uint (uint v)
{{
    uint3 q = hash_uint3_uint3( (uint3)(v, 1, 1) );    
    return (float3)(q.x, q.y, q.z) * UIF;
}}

//---------- Classic hashes used in shaders

float hash_float_float(float p)
{{
    
    float x = sin(p*12.9898)*43758.5453;
    return x - floor(x);
}}

float hash_float_float2(float2 p)
{{
    float x = sin( dot(p, (float2)(12.9898, 78.233)) )*43758.5453;
    return x - floor(x);
}}
"""