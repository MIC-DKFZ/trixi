from __future__ import print_function

import atexit
import fnmatch
import os

import torch

from vislogger.abstractlogger import threaded
from vislogger import ExperimentLogger, PytorchPlotFileLogger
from vislogger.util import name_and_iter_to_filename, update_model


class PytorchExperimentLogger(ExperimentLogger):
    """A single class for logging"""

    def __init__(self, *args, **kwargs):

        super(PytorchExperimentLogger, self).__init__(*args, **kwargs)
        self.plot_logger = PytorchPlotFileLogger(self.img_dir, self.plot_dir)

    def show_images(self, images, name, **kwargs):
        self.plot_logger.show_images(images, name, **kwargs)

    def show_image_grid(self, image, name, **kwargs):
        self.plot_logger.show_image_grid(image, name, **kwargs)

    @staticmethod
    @threaded
    def save_model_static(model, model_dir, name):
        """Stores a model"""

        model_file = os.path.join(model_dir, name)
        torch.save(model.state_dict(), model_file)

    def save_model(self, model, name, n_iter=None, iter_format="{:05d}", prefix=False):

        if n_iter is not None:
            name = name_and_iter_to_filename(name,
                                             n_iter,
                                             ".pth",
                                             iter_format=iter_format,
                                             prefix=prefix)

        if not name.endswith(".pth"):
            name += ".pth"

        self.save_model_static(model=model,
                               model_dir=self.checkpoint_dir,
                               name=name)

    @staticmethod
    @threaded
    def load_model_static(model, model_file, exclude_layers=(), warnings=True):

        if os.path.exists(model_file):

            pretrained_dict = torch.load(model_file, map_location=lambda storage, loc: storage)
            update_model(model, pretrained_dict, exclude_layers, warnings)

        else:

            raise IOError("Model file does not exist!")

    def load_model(self, model, name, exclude_layers=(), warnings=True):

        if not name.endswith(".pth"):
            name += ".pth"

        self.load_model_static(model=model,
                               model_file=os.path.join(self.checkpoint_dir, name),
                               exclude_layers=exclude_layers,
                               warnings=warnings)

    @staticmethod
    @threaded
    def save_checkpoint_static(checkpoint_dir, name, **kwargs):
        for key, value in kwargs.items():
            if isinstance(value, torch.nn.Module) or isinstance(value, torch.optim.Optimizer):
                kwargs[key] = value.state_dict()

        checkpoint_file = os.path.join(checkpoint_dir, name)

        torch.save(kwargs, checkpoint_file)

    def save_checkpoint(self, name, n_iter=None, iter_format="{:05d}", prefix=False, **kwargs):

        if n_iter is not None:
            name = name_and_iter_to_filename(name,
                                             n_iter,
                                             ".pth.tar",
                                             iter_format=iter_format,
                                             prefix=prefix)

        if not name.endswith(".pth.tar"):
            name += ".pth.tar"

        self.save_checkpoint_static(self.checkpoint_dir, name=name, **kwargs)

    @staticmethod
    def load_checkpoint_static(checkpoint_file, exclude_layer_dict=None, warnings=True, **kwargs):

        if exclude_layer_dict is None:
            exclude_layer_dict = {}

        checkpoint = torch.load(checkpoint_file, map_location=lambda storage, loc: storage)

        for key, value in kwargs.items():
            if key in checkpoint:
                if isinstance(value, torch.nn.Module) or isinstance(value, torch.optim.Optimizer):
                    exclude_layers = exclude_layer_dict.get(key, [])
                    update_model(value, checkpoint[key], exclude_layers, warnings)
                else:
                    kwargs[key] = checkpoint[key]

        return kwargs

    def load_checkpoint(self, name, exclude_layer_dict=None, warnings=True, **kwargs):

        if not name.endswith(".pth.tar"):
            name += ".pth.tar"

        checkpoint_file = os.path.join(self.checkpoint_dir, name)
        return self.load_checkpoint_static(checkpoint_file=checkpoint_file,
                                           exclude_layer_dict=exclude_layer_dict,
                                           warnings=warnings,
                                           **kwargs)

    def save_at_exit(self, name="checkpoint_end", **kwargs):

        if not name.endswith(".pth.tar"):
            name += ".pth.tar"

        def save_fnc():
            self.save_checkpoint(name, **kwargs)
            print("Checkpoint saved securely... =)")

        atexit.register(save_fnc)

    def get_save_checkpoint_fn(self, name="checkpoint", **kwargs):

        def save_fnc(n_iter, iter_format="{:05d}", prefix=False):
            self.save_checkpoint(name=name,
                                 n_iter=n_iter,
                                 iter_format=iter_format,
                                 prefix=prefix,
                                 **kwargs)
        return save_fnc

    @staticmethod
    def load_last_checkpoint_static(dir_, name=None, **kwargs):

        if name is None:
            name = "*checkpoint*.pth.tar"

        checkpoint_files = []

        for root, dirs, files in os.walk(dir_):
            for filename in fnmatch.filter(files, name):
                checkpoint_file = os.path.join(root, filename)
                checkpoint_files.append(checkpoint_file)

        if len(checkpoint_files) == 0:
            return None

        last_file = sorted(checkpoint_files, reverse=True)[0]

        return PytorchExperimentLogger.load_checkpoint_static(last_file, **kwargs)

    def load_last_checkpoint(self, **kwargs):
        return self.load_last_checkpoint_static(self.checkpoint_dir, **kwargs)


    def print(self, *args):
        self.file_logger.print(*args)