import atexit
import fnmatch
import os

import torch
from torch.autograd import Variable
from torchvision.utils import save_image

from vislogger.numpyplotlogger import NumpyPlotLogger
from vislogger.util import name_and_iter_to_filename


class PytorchPlotLogger(NumpyPlotLogger):
    """
    Visual logger, inherits the NumpyPlotLogger and plots/ logs pytorch tensors and variables as files on the local
    file system.
    """

    def __init__(self, *args, **kwargs):
        super(PytorchPlotLogger, self).__init__(*args, **kwargs)

    def process_params(self, f, *args, **kwargs):
        """
        Inherited "decorator": convert Pytorch variables and Tensors to numpy arrays
        """

        ### convert args
        args = (a.cpu().numpy() if torch.is_tensor(a) else a for a in args)
        args = (a.data.cpu().numpy() if isinstance(a, Variable) else a for a in args)

        ### convert kwargs
        for key, data in kwargs.items():
            if isinstance(data, Variable):
                kwargs[key] = data.data.cpu().numpy()
            elif torch.is_tensor(data):
                kwargs[key] = data.cpu().numpy()

        return f(self, *args, **kwargs)

    @staticmethod
    def store_image_static(image_dir, tensor, name, n_iter=None, prefix=False, iter_format="{:05d}", normalize=True):
        """Stores an image"""

        img_name = name

        if n_iter is not None:
            img_name = name_and_iter_to_filename(img_name, n_iter, ".png", iter_format=iter_format,
                                                                   prefix=prefix)

        img_file = os.path.join(image_dir, img_name)
        save_image(tensor, img_file, normalize=normalize)

    def store_image(self, tensor, name, n_iter=None, prefix=False, iter_format="{:05d}", normalize=True):
        """Stores an image"""
        PytorchPlotLogger.store_image_static(self.image_dir, tensor=tensor, name=name, n_iter=n_iter,
                                             prefix=prefix, iter_format=iter_format, normalize=normalize)

    @staticmethod
    def store_images_static(image_dir, tensors, n_iter=None, prefix=False, iter_format="{:05d}", normalize=True):
        assert isinstance(tensors, dict)

        for name, tensor in tensors.items():
            PytorchPlotLogger.store_image_static(image_dir=image_dir, tensor=tensor, name=name, n_iter=n_iter,
                                                 prefix=prefix, iter_format=iter_format, normalize=normalize)

    def store_images(self, tensors, n_iter=None, prefix=False, iter_format="{:05d}", normalize=True):
        assert isinstance(tensors, dict)
        PytorchPlotLogger.store_images_static(self.image_dir, tensors=tensors, n_iter=n_iter, prefix=prefix,
                                              iter_format=iter_format, normalize=normalize)

    @staticmethod
    def store_image_grid_static(image_dir, tensor, name, n_iter=None, prefix=False, iter_format="{:05d}", nrow=8,
                                padding=2, normalize=False, range=None, scale_each=False, pad_value=0):

        img_name = name

        if n_iter is not None:
            img_name = name_and_iter_to_filename(img_name, n_iter, ".png", iter_format=iter_format,
                                                                   prefix=prefix)
        elif not img_name.endswith(".png"):
            img_name = img_name + ".png"

        img_file = os.path.join(image_dir, img_name)
        save_image(tensor, img_file, normalize=normalize, nrow=nrow, padding=padding, range=range,
                   scale_each=scale_each, pad_value=pad_value)

    def store_image_grid(self, tensor, name, n_iter=None, prefix=False, iter_format="{:05d}", nrow=8, padding=2,
                         normalize=False, range=None, scale_each=False, pad_value=0):
        PytorchPlotLogger.store_image_grid_static(self.image_dir, tensor=tensor, name=name, n_iter=n_iter,
                                                  prefix=prefix,
                                                  iter_format=iter_format, nrow=nrow, padding=padding,
                                                  normalize=normalize, range=range, scale_each=scale_each,
                                                  pad_value=pad_value)

    def show_image(self, image, name, n_iter=None, prefix=False, iter_format="{:05d}", **kwargs):
        self.store_image(tensor=image, name=name, n_iter=n_iter, prefix=prefix, iter_format=iter_format)

    def show_images(self, images, name, n_iter=None, prefix=False, iter_format="{:05d}", **kwargs):

        tensors = {}
        for i, img in enumerate(images):
            tensors[name + str(i)] = img

        self.store_images(tensors=tensors, n_iter=n_iter, prefix=prefix, iter_format=iter_format)

    def show_image_grid(self, images, name, n_iter=None, prefix=False, iter_format="{:05d}", nrow=8, padding=2,
                        normalize=False, range=None, scale_each=False, pad_value=0, **kwargs):
        self.store_image_grid(tensor=images, name=name, n_iter=n_iter, prefix=prefix, iter_format=iter_format,
                              nrow=nrow,
                              padding=padding,
                              normalize=normalize, range=range, scale_each=scale_each, pad_value=pad_value)

    @staticmethod
    def store_checkpoint_static(model_dir, name, **kwargs):
        for key, value in kwargs.items():
            if isinstance(value, torch.nn.Module) or isinstance(value, torch.optim.Optimizer):
                kwargs[key] = value.state_dict()

        checkpoint_file = os.path.join(model_dir, name)

        torch.save(kwargs, checkpoint_file)

    def store_checkpoint(self, name, **kwargs):
        PytorchPlotLogger.store_checkpoint_static(self.model_dir, name=name, **kwargs)

    @staticmethod
    def restore_checkpoint(name, **kwargs):
        checkpoint = torch.load(name, map_location=lambda storage, loc: storage)

        for key, value in kwargs.items():
            if key in checkpoint:
                if isinstance(value, torch.nn.Module) or isinstance(value, torch.optim.Optimizer):
                    value.load_state_dict(checkpoint[key])
                else:
                    kwargs[key] = checkpoint[key]

        return kwargs

    def save_at_exit(self, **kwargs):
        filename = "checkpoint_end.pth.tar"

        def save_fnc():
            self.store_checkpoint(filename, **kwargs)
            print("Checkpoint stored securely... =)")

        atexit.register(save_fnc)

    def get_store_checkpoint_fn(self, **kwargs):
        def save_fnc(n_iter, iter_format="{:05d}", prefix=False):
            name = name_and_iter_to_filename(name="checkpoint", n_iter=n_iter, ending=".pth.tar",
                                                  iter_format=iter_format,
                                                  prefix=prefix)
            self.store_checkpoint(name, **kwargs)

        return save_fnc

    @staticmethod
    def restore_lastest_checkpoint(dir, name=None, **kwargs):
        if name is None:
            name = "*checkpoint*.pth.tar"

        checkpoint_files = []

        for root, dirs, files in os.walk(dir):
            for filename in fnmatch.filter(files, name):
                checkpoint_file = os.path.join(root, filename)
                checkpoint_files.append(checkpoint_file)

        lastest_file = sorted(checkpoint_files, reverse=True)[0]

        return PytorchPlotLogger.restore_checkpoint(lastest_file, **kwargs)

    @staticmethod
    def restore_best_checkpoint(dir, **kwargs):
        name = "checkpoint_best.pth.tar"
        checkpoint_file = os.path.join(dir, name)

        if os.path.exists(checkpoint_file):
            return PytorchPlotLogger.restore_checkpoint(checkpoint_file, **kwargs)
        else:
            return PytorchPlotLogger.restore_lastest_checkpoint(dir=dir)

    def get_log_dict(self, file_name, log_to_output=False):
        """Creates new dict, which automatically logs all value changes to the given file"""

        return LogDict(logger=self, file_name=file_name, log_to_output=log_to_output)
