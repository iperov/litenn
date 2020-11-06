class Cacheton:
    """
    Cached classes and vars by hashable arguments
    """
    cachetons = {}
    cachevars = {}
    
    @staticmethod
    def get(cls, *args, **kwargs):
        """
        Get class cached by args/kwargs
        If it does not exist, creates new with *args,**kwargs
        All cached data will be freed with nn.cleanup()
        You can store nn.Tensor in cached data.
        """
        cls_multitons = Cacheton.cachetons.get(cls, None)        
        if cls_multitons is None:
            cls_multitons = Cacheton.cachetons[cls] = {}
        
        key = (args, tuple(kwargs.items()) )  
        
        data = cls_multitons.get(key, None) 
        if data is None:
            data = cls_multitons[key] = cls(*args, **kwargs)
        
        return data
    
    @staticmethod
    def set_var(var_name, value):
        """
        Set data cached by var_name
        All cached data will be freed with nn.cleanup()
        You can store nn.Tensor in cached data.
        """
        Cacheton.cachevars[var_name] = value
        
    @staticmethod
    def get_var(var_name):
        """
        Get data cached by var_name
        All cached data will be freed with nn.cleanup()
        You can store nn.Tensor in cached data.
        """
        return Cacheton.cachevars.get(var_name, None)
        
    @staticmethod
    def _cleanup():
        """
        Free all cached objects
        """
        Cacheton.cachetons = {}
        Cacheton.cachevars = {}