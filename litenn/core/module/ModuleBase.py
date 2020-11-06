import logging
log = logging.getLogger('litenn')

import weakref
import numpy as np
import litenn as nn
from ..Saveable import Saveable

class ModuleBase(Saveable):
    """
    Base class for Module and Optimizer
    """
    def __init__(self, saveables=None):
        self._saveables = saveables

    def get_name(self, top_parent=None):
        """
        Returns name of the Module.

        arguments

            top_parent(None)  Module
                        stop constructing the name on this parent
        """
        # Construct the name walking to the top parent
        name = ""
        module = self
        while module is not None:
            if top_parent is not None and \
               top_parent == module:
                break

            parent_module = module._get_parent_module()
            top_name = parent_module._get_child_name(module) \
                       if parent_module is not None else module.__class__.__name__
            name = f'{top_name}.{name}'

            module = parent_module


        name = name.rstrip('.')
        return name

    def load_state(self, state_dict):
        """
        Load state from dict
        """
        ModuleBase._dump_load_state(self, None, self, state_dict, is_dump=False)

    def dump_state(self):
        """
        returns dict of state
        """
        state_dict = {}
        # Gather all nested variables and their names respecting 'saveables' in Modules
        ModuleBase._dump_load_state(self, None, self, state_dict, is_dump=True)
        return state_dict

    def load(self, filepath):
        """
        load the Module from the file.
        If file extension is '.json' then json format is used, otherwise python pickle format.
        """
        super().load(filepath)

    def save(self, filepath):
        """
        Save the Module to the file.
        If file extension is '.json' then json format is used, otherwise python pickle format.
        """
        super().save(filepath)

    ### INTERNAL METHODS start with _

    def _get_parent_module(self):
        # Get parent module from weakref
        _parent_module_w = getattr(self, '_parent_module_w', None)
        if _parent_module_w is None:
            return None
        return _parent_module_w()

    def _get_child_names(self):
        # Get dict of child names where key is weakref to the object
        child_names = getattr(self, '_child_names', None)
        if child_names is None:
            child_names = self._child_names = {}
        return child_names

    def _get_child_name(self, child_obj):
        # Get name of child_obj
        child_names = self._get_child_names()
        return child_names.get( weakref.ref(child_obj), None )

    def __setattr__(self, var_name, obj):
        # Handling setting attrs to the Module
        super().__setattr__(var_name, obj)

        if var_name[0] == '_':
            module_saveables = getattr(self, '_saveables', None)
            if module_saveables is None or var_name not in module_saveables:
                return
        ModuleBase._setattr(self, obj, var_name)

    @staticmethod
    def _setattr(parent_module, obj, name):
        if isinstance(obj, (nn.Tensor,ModuleBase)  ):
            if isinstance(obj, nn.Tensor):
                if obj._is_reference():
                    raise Exception('You cannot assign Reference Tensor to the Module')
                if obj._is_produced_by_op():
                    raise Exception('You cannot assign Tensor produced by operator to the Module')
                if obj._parent_module_w is not None:
                    raise Exception(f'Tensor {obj.get_name()} is already assigned to the Module.')
            obj._parent_module_w = weakref.ref(parent_module)
            parent_module._get_child_names()[weakref.ref(obj)] = name
        elif isinstance(obj, list):
            if len(obj) == 0:
                raise Exception(f"\n\nYou are setting empty list \n'{name}'\n to the Module \n{parent_module.__class__.__name__}.\n Dynamic list variables are not allowed. You may use dynamic list with _internalname, but it will not be saved.")
            for i,element in enumerate(obj):
                ModuleBase._setattr(parent_module, element, f'{name}[{i}]')
        elif isinstance(obj, dict):
            if len(obj) == 0:
                raise Exception(f"\n\nYou are setting empty dict \n'{name}'\n to the Module \n{parent_module.__class__.__name__}.\n Dynamic dict variables are not allowed. You may use dynamic dict with _internalname, but it will not be saved.")
            for key, element in obj.items():
                ModuleBase._setattr(parent_module, element, f'{name}[{key}]')

        elif isinstance(obj, set):
            raise Exception('You cannot assign sets to the Module')

    @staticmethod
    def _dump_load_state(top_parent, parent_module, obj, state_dict, submodule_name=None, is_dump=False, set_func=None):
        if isinstance(obj, ModuleBase):
            module_saveables = getattr(obj, '_saveables', None)

            for var_name in vars(obj):
                if module_saveables is not None:
                    if var_name not in module_saveables:
                        continue
                else:
                    if var_name[0] == '_':
                        continue
                ModuleBase._dump_load_state(top_parent, obj, getattr(obj, var_name), state_dict, var_name, is_dump, lambda val: setattr(obj, var_name, val))
        elif isinstance(obj, nn.Tensor):
            obj_name = obj.get_name(top_parent)
            if is_dump:
                state_dict[obj_name] = ('np.ndarray', obj.np() )
            else:
                value = state_dict.get(obj_name, None)
                if value is not None:
                    obj_type, value = value
                    if obj_type == 'np.ndarray':
                        if tuple(obj.shape) == value.shape:
                            obj.set(value)
                        else:
                            log.warn(f'Tensor {obj_name} is not loaded, because saved Tensor shape is {value.shape}, but must be {tuple(obj.shape)}')
                    else:
                        log.warn(f'Tensor {obj_name} is not loaded, because saved type is {obj_type}, but must be np.ndarray')
                else:
                    log.warn(f'Tensor {obj_name} is not loaded, because saved data is not found.')

        elif isinstance(obj, list):
            go_inside=True

            obj_name = f'{parent_module.get_name(top_parent)}.{submodule_name}'.lstrip('.')
            len_obj_name = f'{obj_name}__len'
            if is_dump:
                state_dict[len_obj_name] = len(obj)
            else:
                list_len = state_dict.get(len_obj_name, None)
                if list_len is None or len(obj) != list_len:
                    log.warn(f'List {obj_name} is not loaded, because length of list {len(obj)} != saved list len {list_len}')
                    go_inside = False

            if go_inside:
                for i,list_obj in enumerate(obj):
                    ModuleBase._dump_load_state(top_parent, parent_module, list_obj, state_dict, f'{submodule_name}[{i}]', is_dump, lambda val: obj.__setitem__(i,val) )

        elif isinstance(obj, dict):
            for dict_key in obj:
                ModuleBase._dump_load_state(top_parent, parent_module, obj[dict_key], state_dict, f'{submodule_name}[{dict_key}]', is_dump, lambda val: obj.__setitem__(dict_key,val) )
        elif isinstance(obj, (int,float,complex,str)):
            obj_name = f'{parent_module.get_name(top_parent)}.{submodule_name}'.lstrip('.')

            if is_dump:
                state_dict[obj_name] = ('python-var', (obj.__class__.__name__, obj) )
            else:
                value = state_dict.get(obj_name, None)
                if value is not None:
                    obj_type, value = value
                    if obj_type == 'python-var':
                        class_name, value = value
                        if class_name == obj.__class__.__name__:
                            set_func(value)
                        else:
                            log.warn(f'Variable {obj_name} is not loaded, because saved class_name is {class_name}, but must be {obj.__class__.__name__}')
                    else:
                        log.warn(f'Variable {obj_name} is not loaded, because saved type is {obj_type}, but must be python-var')
                else:
                    log.warn(f'Variable {obj_name} is not loaded, because saved data is not found.')

