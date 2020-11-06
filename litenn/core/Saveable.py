import pickle
from pathlib import Path

class Saveable:
    """
    Base class of saveable classes.
    """

    #override
    def dump_state(self):
        """
        return a picklable state_dict that should be saved/loaded
        """
        raise NotImplementedError()

    #override
    def load_state(self, state_dict):
        """
        Apply data from state_dict
        """
        raise NotImplementedError()

    #override
    def save(self, filepath):
        """
        Saves data acquired from .dump_state() to filepath
        The data will be stored in pickle format with numpy arrays.
        """
        filepath = Path(filepath)
        filepath.write_bytes(pickle.dumps (self.dump_state(), 4))

    #override
    def load(self, filepath):
        """
        Loads data and calls load_state.
        Returns true if success.
        """
        filepath = Path(filepath)
        if filepath.exists():
            self.load_state(pickle.loads(filepath.read_bytes()))
            return True
        return False
