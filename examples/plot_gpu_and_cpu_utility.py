#!/usr/bin/env python
# -*- coding: utf-8 -*-

#
# Requirements:
#   pip install psutil gputil
#

import numpy as np
from time import sleep
import GPUtil
import psutil
from vislogger import NumpyVisdomLogger as Nvl
nvl = Nvl(name="my_environment")

WINDOW_LEN = 250

data_gpu = [0,0,0]
data_gpu_mean = [0,0,0]
data_cpu = [0,0,0]

ctr = 3
while True:
    sleep(0.2)

    # GPU 0 utility
    load = GPUtil.getGPUs()[0].load # select GPU 0 or another one
    data_gpu.append(load)
    data_gpu_mean.append(np.mean(np.array(data_gpu)[-20:]))
    data_gpu = data_gpu[-WINDOW_LEN:]
    data_gpu_mean = data_gpu_mean[-WINDOW_LEN:]

    x_1 = np.array(range(len(data_gpu)))
    x_2 = np.array(range(len(data_gpu_mean)))

    nvl.show_lineplot(y_vals=np.vstack((np.array(data_gpu), np.array(data_gpu_mean))).T,
                      x_vals=np.vstack((x_1, x_2)).T,
                      name="GPU Utility",
                      opts=dict(title="GPU_0 Util", legend=["Current", "Mean"], width=900))

    # CPU Utility
    cpu_util = np.mean(psutil.cpu_percent(interval=1, percpu=True)) / 100.
    mem_util = psutil.virtual_memory().percent / 100.
    data_cpu.append(cpu_util)
    data_cpu = data_cpu[-WINDOW_LEN:]
    data_cpu[0] = 0  # min
    data_cpu[1] = 1  # max
    nvl.show_lineplot( y_vals=np.array(data_cpu),
                        x_vals=np.array(range(len(data_cpu))),
                        name="CPU Utility",
                        opts=dict(title="CPU Util", width=900)
                        )

    ctr += 1
