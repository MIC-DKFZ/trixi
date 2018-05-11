# -*- coding: utf-8 -*-

import atexit
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

        atexit.register(pynvml.nvmlShutdown)

    def memory_usage(self, device_index=0):

        memory_info = pynvml.nvmlDeviceGetMemoryInfo(
            self._device_handles[device_index])
        return memory_info.used / 1e6, memory_info.total / 1e6

    def memory_usages(self):

        usages = []
        for i in range(self.number_of_devices):
            usages.append(self.memory_usage(i))
        return usages

    def gpu_usage(self, device_index=0):

        return pynvml.nvmlDeviceGetUtilizationRates(
            self._device_handles[device_index]).gpu

    def gpu_usages(self):

        ut = []
        for i in range(self.number_of_devices):
            ut.append(self.gpu_usage(i))
        return ut

    def temperature(self, device_index=0):

        return pynvml.nvmlDeviceGetTemperature(
            self._device_handles[device_index], pynvml.NVML_TEMPERATURE_GPU)

    def temperatures(self):

        temp = []
        for i in range(self.number_of_devices):
            temp.append(self.temperature(i))
        return temp

    def power_usage(self, device_index=0):

        handle = self._device_handles[device_index]
        used = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.
        total = pynvml.nvmlDeviceGetPowerManagementLimit(handle) / 1000.
        return used, total

    def power_usages(self):

        pwr = []
        for i in range(self.number_of_devices):
            pwr.append(self.power_usage(i))
        return pwr

    def fan_usage(self, device_index=0):

        return pynvml.nvmlDeviceGetFanSpeed(self._device_handles[device_index])

    def fan_usages(self):

        fan = []
        for i in range(self.number_of_devices):
            fan.append(self.fan_usage(i))
        return fan