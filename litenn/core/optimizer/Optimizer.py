import numpy as np
from ..Saveable import Saveable
import litenn as nn
import litenn.core as nc
from litenn.core import CLKernelHelper as ph

class Optimizer(nc.ModuleBase):
    """
    Base class for optimizers.
    
    arguments

        tensor_list     list of tensors which should be optimized

        lr(0.001)       learning rate
        
        lr_decay(0.0)   learning rate decay
        
        lr_dropout(0.0) learning rate dropout.
                        [0.0 .. 1.0) probability
                        typical value is 0.7
                        
        clipnorm(0.0)   Clip the gradient if the L2norm exceeds clipnorm.
        
        
        saveables(None) list of strings of variables that should be saved
                        for example: 
                            .accs = {}     # dict of accumulators
                            .yourvar = 0   #
                        saveables=['accs','yourvar']
                        
                        You should not save optimizer options
                        such as learning rate, momentum, rho, etc...
    """

    def __init__(self, _tensors_list, lr=0.001, lr_decay=0.0, lr_dropout=0.0, clipnorm=0.0, saveables=None):
        if _tensors_list is None:
            raise ValueError('_tensors_list must be specified.')

        # Check tensors
        for i,t1 in enumerate(_tensors_list):
            for j,t2 in enumerate(_tensors_list):
                if i == j:
                    continue
                    
                if t1 == t2:
                    raise ValueError(f'Tensor {t1.get_name()} has duplicate in a list.')
            
                if t1.get_name() == t2.get_name():
                    raise ValueError(f"""
Different tensors with the same name '{t1.get_name()}' indexes {i} {j}.
If you are using the same module twice, possible solution:

class DecoderA(Decoder): pass
class DecoderB(Decoder): pass
""")  
        
        for t in _tensors_list:
            if t._is_reference():
                raise ValueError(f'Tensor {t.get_name()} is reference, and cannot be trained in optimizer.')
            if t._is_grad():
                raise ValueError(f'Tensor {t.get_name()} is gradient tensor, and cannot be trained in optimizer.')
            if t._is_trainable():
                raise ValueError(f'Tensor {t.get_name()} is already used by another optimizer.')
            if t._get_gradfns() is not None:
                raise Exception(f'Tensor {t.get_name()} is produced by operator and cannot be trained in optimizer.')
            t._set_trainable(True)
            
        # tensors_list starts with _ will be ignored by nn.Module
        self._tensors_list = _tensors_list
        self.lr = lr
        self.lr_decay = lr_decay
        self.lr_dropout = lr_dropout
        self.clipnorm = clipnorm
        self.iteration = 0
        
        if saveables is None:
            saveables = []
        super().__init__(saveables+['iteration'])
        
    def get_trainable_tensors(self): return self._tensors_list
    def get_iteration(self):
        """
        returns current iteration number.
        """
        return self.iteration
    
    def set_iteration(self, iteration):
        """
        override current iteration number.
        """
        self.iteration = iteration        
        
    def zero_grad(self):
        """
        Zeroes gradient of tensors
        """
        for t in self._tensors_list:
            t.zero_grad()
            
    def print_no_grad(self):
        """
        Print those .get_trainable_tensors() that have no gradient.
        
        Should be used after 
        
         .backward 
         
        and before 
        
         optimizer.step()
         
        Can be used for debugging purposes. 
        Freezed tensors under nn.optimizer.freeze()
        will also be printed
        """
        for t in self.get_trainable_tensors():
            if not t.has_grad():
                print(f'Tensor has no gradient : {t.get_name()}')
            
            
    def step(self, multi_gpu_step=False):
        """
        Perform tensors optimization using their gradients.

         multi_gpu_step(False)  perform average multigpu tensors
        """

        if self.clipnorm != 0.0:
            # Calculate L2Norm
            sums = []
            for t in self._tensors_list:
                if not t.has_grad():
                    continue
                sums.append( nn.square(t.get_grad()).sum().np()[0] )
            l2norm = np.sqrt(np.sum(sums))

            if l2norm > self.clipnorm:
                # l2norm exceeds clipnorm
                # apply in-place modifier
                mod = self.clipnorm / l2norm
                for t in self._tensors_list:
                    if not t.has_grad():
                        continue
                    nc.op.mul_const(t.get_grad(), mod, output_t=t.get_grad())

        self._on_step()

        if multi_gpu_step and \
           len(nn.devices.get_current()) > 1:
            # MultiGPU step : average tensor's values
            for t in self._tensors_list:
                t.set( np.mean( [ t.np(i) for i in range(t.np_count()) ] , 0 ) )

        self.iteration += 1
        
    #override
    def _on_step(self):
        raise NotImplementedError()

    def _get_lr_kernel_common_text(self):
        """
        Returns common kernel text.
        Should be placed before __kernel.
        """        
        return f"""
{ph.include_hash()}
#define LR_DROPOUT {'1' if self.lr_dropout != 0.0 else '0'}
#define LR_DECAY {'1' if self.lr_decay != 0.0 else '0'}
"""
        
    def _get_lr_kernel_args_text(self):
        """
        Returns kernel argument definitions.
        """        
        return ", uint iteration, uint seed"
        
    def _get_lr_kernel_args(self):
        """
        Returns kernel arguments to be passed to kernel.run()
        """
        return [np.uint32(self.iteration), np.uint32(np.random.randint(2147483648))]
        
    def _get_lr_kernel_text(self):
        """
        Returns kernel text.
        Provides 
         float lr;
        already processed by decay and dropout.        
        """
        
        return f"""
float lr = {self.lr};

#if LR_DECAY == 1
    lr *= 1.0 / ( 1.0 + {self.lr_decay} * iteration );
#endif

#if LR_DROPOUT == 1
    if ( hash_float_uint(gid+seed) <= {self.lr_dropout} )
        lr = 0;
#endif
    
"""