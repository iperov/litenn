from typing import List
from collections import Iterable
from .api import OpenCL as CL
from .CLDevice import CLDevice
from .CLBuffer import CLBuffer
from .CLShallowMode import CLShallowMode


class devices:
    """
    Static class to operate current devices.
    """

    _current_devices = None
    _all_devices = None
        
    @staticmethod
    def _shallow_mode():
        """
        Kernels will not be executed on all current devices.
        Memory allocation will raise an exception.
        Usage

            with devices._shallow_mode():
                ...
                your code here
                ...
        """
        return CLShallowMode(devices.get_current())

    @staticmethod
    def get_current() -> List[CLDevice]:
        if devices._current_devices is None:
            best_device = devices.get_best()
            if best_device is None:
                raise Exception("No valid OpenCL devices found.")       
            devices.set_current(best_device)
        return devices._current_devices

    @staticmethod
    def set_current(device_or_list):
        """
        Change current devices.

        arguments

            device_or_list  is a CLDevice or list of CLDevice
        """
        if CLBuffer._object_count != 0:
            raise Exception('You cannot set current devices while any CLBuffer objects exists.')

        devices.cleanup()

        if isinstance(device_or_list, Iterable):
            device_or_list = tuple(device_or_list)
        if not isinstance(device_or_list, tuple):
            device_or_list = (device_or_list,)
        if len(device_or_list) == 0:
            raise ValueError('device_or_list cannot be empty')
        if not all (isinstance(device, CLDevice) for device in device_or_list):
            raise ValueError(f'argument {device.__class__} should be CLDevice')

        print(f'Using { ", ".join (device.name for device in device_or_list) }')
        
        devices._current_devices = device_or_list

    @staticmethod
    def cleanup():
        """
        Frees resources of all current devices. CLBuffer object count must be zero.
        """
        if CLBuffer._object_count != 0:
            raise Exception(f'Unable to cleanup while {CLBuffer._object_count} CLBuffer objects exist.')

        if devices._current_devices is not None:
            for device in devices._current_devices:
                device._cleanup()

    @staticmethod
    def wait():
        """
        Wait to finish all pending operations on all devices.
        """
        for device in devices.get_current():
            device.wait()

    @staticmethod
    def get_all() -> List[CLDevice]:
        """
        Returns list of all available Devices.
        """
        if devices._all_devices is None:
            ar = devices._all_devices = []
            idx = 0
            for is_cpu, DEVICE_TYPE in [ (False, CL.DEVICE_TYPE.GPU | CL.DEVICE_TYPE.ACCELERATOR),
                                         (True, CL.DEVICE_TYPE.CPU)  ]:
              for platform in CL.GetPlatforms():
                for device in CL.GetDeviceIDs(platform, DEVICE_TYPE=DEVICE_TYPE):
                    ar.append ( CLDevice(device=device,
                                         index=idx,
                                         name=device.name,
                                         global_mem_size=device.global_mem_size,
                                         is_cpu=is_cpu) )
                    idx += 1
        return devices._all_devices
    
    @staticmethod
    def get_by_idx(index) -> CLDevice:
        """
        Returns Device by index.
        """
        for dev in devices.get_all():
            if dev.index == index:
                return dev
        return None
        
        
    @staticmethod
    def get_equal(device) -> List[CLDevice]:
        """
        Returns list of devices the same as device
        """
        device_name = device.name
        result = []
        for device in devices.get_all():
            if device.name == device_name:
                result.append (device)
        return result

    @staticmethod
    def get_best_equal() -> List[CLDevice]:
        """
        Returns list of devices the same as device_get_best()
        """
        return devices.get_equal(devices.get_best())

    @staticmethod
    def get_worst_equal() -> List[CLDevice]:
        """
        Returns list of devices the same as device_get_worst()
        """
        return devices.get_equal(devices.get_worst())

    @staticmethod
    def get_best() -> CLDevice:
        """
        Returns CLDevice with the largest VRAM
        """
        result = None
        idx_mem = 0
        for device in devices.get_all():
            mem = device.global_mem_size
            if mem > idx_mem:
                result = device
                idx_mem = mem
        return result

    @staticmethod
    def get_worst() -> CLDevice:
        """
        Returns CLDevice with the smallest VRAM
        """
        result = None
        idx_mem = sys.maxsize
        for device in devices.get_all():
            mem = device.global_mem_size
            if mem < idx_mem:
                result = device
                idx_mem = mem
        return result

    @staticmethod
    def ask_to_choose(choose_only_one=False, allow_cpu=True, suggest_best_multi_gpu=False, suggest_all_devices=False):
        """
        Ask user to choose device(s).

        arguments

        choose_only_one(False)         user can choose only one device

        allow_cpu(True)                user will see CPU devices in a list

        suggest_best_multi_gpu(False)  default suggestion is multiple GPUs with the largest amount of VRAM

        suggest_all_devices(False)     default suggestion is all devices

        default suggestion - one GPU with the largest amount of VRAM

        Example of console:
        Choose one or several device idxs (separated by comma).

        [0] : GeForce RTX 2080 Ti
        [1] : GeForce GTX 750 Ti

        [0] Which device indexes to choose? :
        """
        list_devices = devices.get_all()
        if len(list_devices) == 0:
            return []

        #all_devices_indexes = [device.index for device in devices]

        if choose_only_one:
            suggest_best_multi_gpu = False
            suggest_all_devices = False

        if suggest_all_devices:
            suggestion_devices = devices.get_all()
        elif suggest_best_multi_gpu:
            suggestion_devices = devices.get_best_equal()
        else:
            suggestion_devices = [devices.get_best()]

        #best_device_indexes = ",".join([str(x) for x in best_device_indexes])

        print("")
        if choose_only_one:
            print ("Choose one device index.")
        else:
            print ("Choose one or several device idxs (separated by comma).")
        print ("")

        if not allow_cpu:
            list_devices = [device for device in list_devices if not device.is_cpu]
            suggestion_devices = [device for device in suggestion_devices if not device.is_cpu]

        if len(list_devices) == 0:
            print("You have no valid devices.")
            return []

        list_devices_indexes = [ device.index for device in list_devices ]

        for device in list_devices:
            print (f"  [{device.index}] : {device.name}")

        suggestion_devices_indexes_str = ",".join(str(device.index) for device in suggestion_devices)

        print ("")
        while True:
            try:
                if choose_only_one:
                    choosed_idxs = _input_str("Which device index to choose?", suggestion_devices_indexes_str)
                else:
                    choosed_idxs = _input_str("Which device indexes to choose?", suggestion_devices_indexes_str)

                choosed_idxs = [ int(x) for x in choosed_idxs.split(',') ]
                if len(choosed_idxs) > len(set(choosed_idxs)):
                    continue

                if choose_only_one:
                    if len(choosed_idxs) == 1:
                        break
                else:
                    if all( [idx in list_devices_indexes for idx in choosed_idxs] ):
                        break
            except:
                continue

        print ("")

        return [ device for idx in choosed_idxs for device in list_devices if idx == device.index ]


def _input_str(s, default_value=None, valid_list=None, show_default_value=True, help_message=None):
    if show_default_value and default_value is not None:
        s = f"[{default_value}] {s}"

    if valid_list is not None or \
        help_message is not None:
        s += " ("

    if valid_list is not None:
        s += " " + "/".join(valid_list)

    if help_message is not None:
        s += " ?:help"

    if valid_list is not None or \
        help_message is not None:
        s += " )"

    s += " : "


    while True:
        try:
            inp = input(s)

            if len(inp) == 0:
                if default_value is None:
                    print("")
                    return None
                result = default_value
                break

            if help_message is not None and inp == '?':
                print(help_message)
                continue

            if valid_list is not None:
                if inp.lower() in valid_list:
                    result = inp.lower()
                    break
                if inp in valid_list:
                    result = inp
                    break
                continue

            result = inp
            break
        except:
            result = default_value
            break

    print(result)
    return result