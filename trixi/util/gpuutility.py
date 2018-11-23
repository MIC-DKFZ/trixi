from time import sleep
import threading

import numpy as np
import GPUtil
import psutil

from trixi.logger import NumpyVisdomLogger as Nvl


def plot_gpu_and_cpu_utility(nvl=None, gpu_nr=0):
    """
    Plot GPU, CPU and memory (RAM) utility in an automatically updating line graph. The plot is updated in its own
    daemon thread. I automatically gets killed when the program calling this function gets killed.

    TODO:
    - write tests
    - catch case if no Nvidia GPU available

    Args:
        nvl: Pass a NumpyVisdomLogger where to plot the utility. If none is passed a new one will be created.
        gpu_nr: Number of the GPU for which to plot the utility.

    Returns:
        Void
    """

    def async_plotting_thread(nvl, gpu_nr):
        WINDOW_LEN = 250
        UPDATE_INTERVAL = 0.2

        history_gpu = np.zeros(WINDOW_LEN)
        history_gpu_mean = np.zeros(WINDOW_LEN)
        history_cpu = np.zeros(WINDOW_LEN)
        history_mem = np.zeros(WINDOW_LEN)

        while True:
            sleep(UPDATE_INTERVAL)

            # GPU utility
            gpu_load = GPUtil.getGPUs()[gpu_nr].load
            history_gpu = np.append(history_gpu, gpu_load)
            history_gpu_mean = np.append(history_gpu_mean, history_gpu[-20:].mean())
            history_gpu = history_gpu[-WINDOW_LEN:]  # keep window size
            history_gpu_mean = history_gpu_mean[-WINDOW_LEN:]

            x_idx = np.array(range(len(history_gpu)))
            nvl.show_lineplot(y_vals=np.vstack((history_gpu, history_gpu_mean)).T,
                              x_vals=np.vstack((x_idx, x_idx)).T,
                              name="GPU Utility",
                              opts=dict(title="GPU Utility", legend=["Current", "Mean"], width=600))

            # CPU Utility
            cpu_util = np.mean(psutil.cpu_percent(interval=1, percpu=True)) / 100.
            history_cpu = np.append(history_cpu, cpu_util)
            history_cpu = history_cpu[-WINDOW_LEN:]
            # Add min/max to properly scale plot
            history_cpu[0] = 0  # min
            history_cpu[1] = 1  # max
            mem_util = psutil.virtual_memory().percent / 100.
            history_mem = np.append(history_mem, mem_util)
            history_mem = history_mem[-WINDOW_LEN:]

            x_idx = np.array(range(len(history_gpu)))
            nvl.show_lineplot(y_vals=np.vstack((history_cpu, history_mem)).T,
                              x_vals=np.vstack((x_idx, x_idx)).T,
                              name="CPU Utility",
                              opts=dict(title="CPU Utility", legend=["CPU", "Memory"], width=600))

    if nvl is None:
        nvl = Nvl(name="GPU_Utility")

    th = threading.Thread(target=async_plotting_thread, args=(nvl, gpu_nr))
    th.daemon = True
    th.start()

