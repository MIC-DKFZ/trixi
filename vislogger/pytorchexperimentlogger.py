from __future__ import print_function

import datetime
import json
import os

import torch

from vislogger import ExperimentLogger, PytorchPlotLogger
from vislogger.util import name_and_iter_to_filename


class PytorchExperimentLogger(ExperimentLogger):
    """A single class for logging"""

    def __init__(self, *args, **kwargs):

        super(PytorchExperimentLogger, self).__init__(*args, **kwargs)
        self.plot_logger = PytorchPlotLogger(self.img_dir, self.plot_dir)

    def show_images(self, images, name, **kwargs):
        self.plot_logger.show_images(images, name, **kwargs)

    def show_image_grid(self, image, name, **kwargs):
        self.plot_logger.show_image_grid(image, name, **kwargs)

    @staticmethod
    def save_model_static(model_dir, model, name, n_iter=None, prefix=False, iter_format="{:05d}"):
        """Stores a model"""

        if n_iter is not None:
            name = name_and_iter_to_filename(name,
                                             n_iter,
                                             ".pth",
                                             iter_format=iter_format,
                                             prefix=prefix)

        model_file = os.path.join(model_dir, name)
        torch.save(model.state_dict(), model_file)

    def save_model(self, model, name, n_iter=None, prefix=False, iter_format="{:05d}"):

        self.save_model_static(self.model_dir,
                               model=model,
                               name=name,
                               n_iter=n_iter,
                               prefix=prefix,
                               iter_format=iter_format)

    @staticmethod
    def load_model_static(model, model_file, exclude_layers=None):

        if os.path.exists(model_file):

            if exclude_layers is None:
                exclude_layers = []

            # also allow loading of partially pretrained net
            pretrained_dict = torch.load(model_file, map_location=lambda storage, loc: storage)
            model_dict = model.state_dict()

            # 1. filter out unnecessary keys
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if
                               k in model_dict and k not in exclude_layers}
            # 2. overwrite entries in the existing state dict
            model_dict.update(pretrained_dict)
            # 3. load the new state dict
            model.load_state_dict(model_dict)

        else:

            raise IOError("Model file does not exist!")

    def load_model(self, model, name, exclude_layers=None):

        if not name.endswith(".pth"):
            name += ".pth"

        self.load_model_static(model, os.path.join(self.model_dir, name), exclude_layers)

    def save_checkpoint(self):
        raise NotImplementedError

    def load_checkpoint(self):
        raise NotImplementedError