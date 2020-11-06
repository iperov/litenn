import logging
import weakref
import numpy as np
import litenn as nn
import litenn.core as nc

class Module(nc.ModuleBase):
    """
    Module is a base class for layers/models
    which contain saveable parameters (python variables or Tensor's)

    also provides Tensor operations by calling the module.

    Types of variables that can be saved:
        Module, Tensor
        int, float, string
        nested lists or dicts of above types.

    - ! variable name starts with _ is ignored by Module, unless it is in saveables !

    - ! Lists must not be dynamic !
    It's mean before Module loading, list must be initialized to corresponding size,
    otherwise an Exception will be occured on access.

    - ! All lists/dicts must be initialized before set to the Module !
      Example:

        ar = []
        ar.append( someModule() )
        self.ar = ar

      Setting empty list will raise an exception.


    Module() constructor arguments:

        saveables(None)     List of strings
                            specify which variable names (Tensor and python variables) can be saved.
                            if None - all supported variables will be saved.
                            
        trainables(None)    List of strings
                            specify which variable names (Tensors) 
                            should be returned in trainables() method.
                            if None - they are the same as saveables

    calling Module() constructor can be omitted.
    """

    def __init__(self, saveables=None, trainables=None):
        super().__init__(saveables=saveables)
        self._trainables = trainables

        
    def forward(self, *args, **kwargs):
        """
        Define your operations here.
        """
        raise NotImplementedError()

    def shallow_forward(self, *args, **kwargs):
        """
        same as forward() but without kernel execution and physical memory allocation.
        You can use it to compute output shape of the Module.
        """
        with nn.devices._shallow_mode():
            return self.forward(*args, **kwargs)
            
    def is_training(self):
        """
        returns True if Module is in training mode
        can be changed by set_training() to the most parent Module
        """
        return getattr(self, '_tr', False)

    def set_training(self, is_training):
        """
        set training flag to this and all submodules.
        This flag changes behaviour of forward() of some layers
        such as Dropout, BatchNormalization.
        """
        Module._nested_set_training(self, is_training)
  
  
    def trainables(self):
        """
        Returns list of trainable Tensors from this module and nested modules.
        
        This list is constructed from:
        
         1) trainables
         2) saveables, if trainables is None
         3) all tensors is saveables is None
        
        This list can be used in Optimizers.

        You can make your own function which will return a bunch of tensors to use in optimizer.
        for example

            get_stage_tensors(stage=1)
                return self.conv1.trainables()+[self.dense1.weight, self.dense2.weight]
                
        also you can freeze specific weights directly in forward() 
        using nn.optimizer.freeze
        """
        out_list = []

        module_trainables = getattr(self, '_trainables', None) 
        if module_trainables is None:
            module_trainables = getattr(self, '_saveables', None)
                             
        for var_name in vars(self):
            if module_trainables is not None:
                if var_name not in module_trainables:
                    continue
            else:
                if var_name[0] == '_':
                    continue
            Module._get_trainable_tensors(getattr(self, var_name), out_list)
        return out_list
            
    ### INTERNAL METHODS start with _
    
    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)
    

    @staticmethod
    def _get_trainable_tensors(obj, out_list):
        if isinstance(obj, Module):
            out_list.extend(obj.trainables())
        elif isinstance(obj, nn.Tensor):
            out_list.append(obj)
        elif isinstance(obj, list):
            for list_obj in obj:
                Module._get_trainable_tensors(list_obj, out_list)
        elif isinstance(obj, dict):
            for dict_key in obj:
                Module._get_trainable_tensors(obj[dict_key], out_list)

    
    @staticmethod
    def _nested_set_training(obj, is_training):
        if isinstance(obj, Module):
            obj._tr = is_training
            for var_name in vars(obj):
                Module._nested_set_training(getattr(obj, var_name), is_training )
        elif isinstance(obj, list):
            for i, list_obj in enumerate(obj):
                Module._nested_set_training(list_obj, is_training )
        elif isinstance(obj, dict):
             for dict_key, dict_obj in obj.items():
                Module._nested_set_training(dict_obj, is_training )

