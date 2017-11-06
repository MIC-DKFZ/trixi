# -*- coding: utf-8 -*-

import pynvml

class GpuMonitor(object):
    """Utility class to monitor GPU usage.

    Uses pynvml (package name nvidia-ml-py). The current pypi version doesn't
    support Python 3. If you're using Python 3, try installing from Github
    directly via
    `pip install git+https://github.com/jonsafari/nvidia-ml-py@master`.
    The reason we're using a class is that NVML has to be initialized, and it
    seems to be cleaner not to initialize and shut down NVML for each request
    (that's just a guess, could be that it's actually designed to be used that
    way).
    """

    def __init__(self):

        super(GpuMonitor, self).__init__()

        pynvml.nvmlInit()

        self.number_of_devices = pynvml.nvmlDeviceGetCount()
        self._device_handles = [pynvml.nvmlDeviceGetHandleByIndex(i)\
                                for i in range(self.number_of_devices)]

    def memory_usage(self, device_index=0):

        memory_info = pynvml.nvmlDeviceGetMemoryInfo(
            self._device_handles[device_index])
        return memory_info.used, memory_info.total

    def memory_usages(self):

        usages = []
        for i in range(self.number_of_devices):
            usages.append(self.memory_usage(i))
        return usages

    def __del__(self):

        pynvml.nvmlShutdown()
        super(GpuMonitor, self).__del__()