import atexit
from collections import defaultdict

import numpy as np
import torch
import torchvision.utils as vutils
from tensorboardX import SummaryWriter

from vislogger import AbstractLogger


class TensorboardLogger(AbstractLogger):
    """Abstract interface for visual logger."""

    def __init__(self, target_dir):
        self.writer = SummaryWriter(target_dir)

        self.val_dict = defaultdict(int)

        atexit.register(self.writer.close)



    def show_image(self, image, name="Image", **kwargs):
        """Abstract method which should handle and somehow log/ store an image"""
        self.writer.add_image(name, image)

    def show_value(self, value, name="Value", counter=None, tag=None, **kwargs):
        """Abstract method which should handle and somehow log/ store a value"""

        if tag is None:
            key = name + "-" + name
        else:
            key = tag + "-" + name

        if counter is not None:
            self.val_dict[key] = counter

        if tag is not None:
            self.writer.add_scalars(tag, {name: value}, global_step=self.val_dict[key])
        else:
            self.writer.add_scalar(name, value, global_step=self.val_dict[key])

        self.val_dict[key] += 1

    def show_text(self, text, name="Text", **kwargs):
        """Abstract method which should handle and somehow log/ store a text"""
        self.writer.add_text(name, text)

    def show_image_grid(self, tensor, name="Images", image_args=None, **kwargs):
        if isinstance(tensor, np.ndarray):
            tensor = torch.from_numpy(tensor)
        x = vutils.make_grid(tensor, **image_args)
        self.writer.add_image(name, x)

    def show_histogramm(self, np_array, name="Histogram", **kwargs):
        """Abstract method which should handle and somehow log/ store a barplot"""
        self.writer.add_histogram(name, np_array)

    def plot_model_structure(self, *args, **kwargs):
        """Abstract method which should handle and somehow log/ store a lineplot"""
        raise NotImplementedError()
