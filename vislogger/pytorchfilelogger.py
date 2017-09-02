import atexit
import fnmatch
import os

import torch
from torch.autograd import Variable
from torchvision.utils import save_image

from vislogger.numpyfilelogger import NumpyFileLogger


class LogDict(dict):
    def __init__(self, file_name, logger, log_to_output=False, log_temp_output=True):
        """Initilaizes a new Dict which directly logs value chnages to a given target_file."""

        super(LogDict, self).__init__()

        self.logger = logger
        self.file_name = file_name
        self.logger.add_log_file(self.file_name)
        self.log_to_output = log_to_output
        self.log_temp_output = log_temp_output

    def __setitem__(self, key, item):
        super(LogDict, self).__setitem__(key, item)

        if self.log_temp_output:

            item = item.cpu().numpy() if torch.is_tensor(item) else item
            item = item.data.cpu().numpy() if isinstance(item, Variable) else item

            self.logger.log_to(self.file_name, "%s : %s" % (key, item), log_to_output=self.log_to_output)

    def __getattr__(self, attr):
        return self.get(attr)

    def __setattr__(self, key, value):
        self.__setitem__(key, value)

    def __delitem__(self, key):
        super(LogDict, self).__delitem__(key)
        del self.__dict__[key]

    def log_content(self):
        """Logs the current content of the dict to the output file as a whole."""
        self.logger.log_to(self.file_name, str(self), log_to_output=self.log_to_output)


class PytorchFileLogger(NumpyFileLogger):
    """
    Visual logger, inherits the NumpyFileLogger and plots/ logs pytorch tensors and variables as files on the local
    file system.
    """

    def __init__(self, *args, **kwargs):
        super(PytorchFileLogger, self).__init__(*args, **kwargs)

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

    def name_and_iter_to_filename(self, name, n_iter, ending, iter_format="%05d", prefix=False):
        iter_str = iter_format % n_iter
        if prefix:
            name = iter_str + "_" + name + ending
        else:
            name = name + "_" + iter_str + ending

        return name

    def store_image(self, tensor, name, n_iter=None, prefix=False, iter_format="%05d", normalize=True):
        """Stores an image"""

        img_name = name

        if n_iter is not None:
            img_name = self.name_and_iter_to_filename(img_name, n_iter, ".png", iter_format=iter_format, prefix=prefix)

        img_file = os.path.join(self.image_dir, img_name)
        save_image(tensor, img_file, normalize=normalize)

    def store_images(self, tensors, n_iter=None, prefix=False, iter_format="%05d", normalize=True):
        assert isinstance(tensors, dict)

        for name, tensor in tensors.items():
            self.store_image(tensor=tensor, name=name, n_iter=n_iter, prefix=prefix, iter_format=iter_format,
                             normalize=normalize)

    def store_image_grid(self, tensor, name, n_iter=None, prefix=False, iter_format="%05d", nrow=8, padding=2,
                         normalize=False, range=None, scale_each=False, pad_value=0):

        img_name = name

        if n_iter is not None:
            img_name = self.name_and_iter_to_filename(img_name, n_iter, ".png", iter_format=iter_format, prefix=prefix)
        elif not img_name.endswith(".png"):
            img_name = img_name + ".png"

        img_file = os.path.join(self.image_dir, img_name)
        save_image(tensor, img_file, normalize=normalize, nrow=nrow, padding=padding, range=range,
                   scale_each=scale_each, pad_value=pad_value)

    def show_image(self, image, name, n_iter=None, prefix=False, iter_format="%05d", **kwargs):
        self.store_image(tensor=image, name=name, n_iter=n_iter, prefix=prefix, iter_format=iter_format)

    def show_images(self, images, name, n_iter=None, prefix=False, iter_format="%05d", **kwargs):

        tensors = {}
        for i, img in enumerate(images):
            tensors[name + str(i)] = img

        self.store_images(tensors=tensors, n_iter=n_iter, prefix=prefix, iter_format=iter_format)

    def show_image_grid(self, images, name, n_iter=None, prefix=False, iter_format="%05d", nrow=8, padding=2,
                        normalize=False, range=None, scale_each=False, pad_value=0, **kwargs):
        self.store_image_grid(tensor=images, name=name, n_iter=n_iter, prefix=prefix, iter_format=iter_format,
                              nrow=nrow,
                              padding=padding,
                              normalize=normalize, range=range, scale_each=scale_each, pad_value=pad_value)

    def store_model(self, model, name, n_iter=None, prefix=False, iter_format="%05d"):
        """Stores a model"""

        model_name = name

        if n_iter is not None:
            model_name = self.name_and_iter_to_filename(model_name, n_iter, ".pth", iter_format=iter_format,
                                                        prefix=prefix)

        model_file = os.path.join(self.model_dir, model_name)
        torch.save(model.state_dict(), model_file)

    def store_models(self, models, n_iter=None, prefix=False, iter_format="%05d"):
        assert isinstance(models, dict)

        for name, model in models.items():
            self.store_model(model=model, name=name, n_iter=n_iter, prefix=prefix, iter_format=iter_format)

    def load_model(self, model, model_file, exclude_layers=None):
        if os.path.exists(model_file):

            if exclude_layers is None:
                exclude_layers = []

            # also allow loading of partially pretrained net
            pretrained_dict = torch.load(model_file)
            model_dict = model.state_dict()

            # 1. filter out unnecessary keys
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict and k not in exclude_layers}
            # 2. overwrite entries in the existing state dict
            model_dict.update(pretrained_dict)
            # 3. load the new state dict
            model.load_state_dict(model_dict)

        else:
            raise ValueError("Model file does not exisit !")

    def load_model_from_dir(self, model, dir, name, n_iter=None, prefix=False, iter_format="%05d", exclude_layers=None):

        # If not iteration given, take the most recent model
        if n_iter is None:

            model_files = []
            match_name = "*" + name + "*.pth"

            for root, dirs, files in os.walk(dir):
                for filename in fnmatch.filter(files, match_name):
                    model_file = os.path.join(root, filename)
                    model_files.append(model_file)

            model_name = sorted(model_files, reverse=True)[1]

        else:
            model_name = self.name_and_iter_to_filename(name, n_iter, ".pth", iter_format=iter_format, prefix=prefix)

        model_file = os.path.join(dir, model_name)
        self.load_model(model, model_file, exclude_layers=exclude_layers)

    def load_models_from_dir(self, models, dir, n_iter=None, prefix=False, iter_format="%05d", step_size=25,
                             exclude_layers={}):
        assert isinstance(models, dict) and isinstance(exclude_layers, dict)

        for name, model in models.items():
            self.load_model_from_dir(model, dir, name, n_iter=n_iter, prefix=prefix, iter_format=iter_format,
                                     step_size=step_size, exclude_layers=exclude_layers.get(name))

    def store_checkpoint(self, name, **kwargs):
        for key, value in kwargs.items():
            if isinstance(value, torch.nn.Module) or isinstance(value, torch.optim.Optimizer):
                kwargs[key] = value.state_dict()

        checkpoint_file = os.path.join(self.model_dir, name)

        torch.save(kwargs, checkpoint_file)

    def restore_checkpoint(self, name, **kwargs):
        checkpoint = torch.load(name)

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
        def save_fnc(n_iter, iter_format="%05d", prefix=False):
            name = self.name_and_iter_to_filename(name="checkpoint", n_iter=n_iter, ending=".pth.tar",
                                                  iter_format=iter_format,
                                                  prefix=prefix)
            self.store_checkpoint(name, **kwargs)

        return save_fnc

    def restore_lastest_checkpoint(self, dir, name=None, **kwargs):
        if name is None:
            name = "*checkpoint*.pth.tar"

        checkpoint_files = []

        for root, dirs, files in os.walk(dir):
            for filename in fnmatch.filter(files, name):
                checkpoint_file = os.path.join(root, filename)
                checkpoint_files.append(checkpoint_file)

        lastest_file = sorted(checkpoint_files, reverse=True)[0]

        return self.restore_checkpoint(lastest_file, **kwargs)

    def restore_best_checkpoint(self, dir, **kwargs):
        name = "checkpoint_best.pth.tar"
        checkpoint_file = os.path.join(dir, name)

        if os.path.exists(checkpoint_file):
            return self.restore_checkpoint(checkpoint_file, **kwargs)
        else:
            return self.restore_lastest_checkpoint(dir=dir)



    def get_log_dict(self, file_name, log_to_output=False):
        """Creates new dict, which automatically logs all value changes to the given file"""

        return LogDict(logger=self, file_name=file_name, log_to_output=log_to_output)
