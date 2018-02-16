import os

import torch
from torch.autograd import Variable
from torchvision.utils import save_image as tv_save_image

from vislogger.abstractlogger import threaded
from vislogger.numpyplotfilelogger import NumpyPlotFileLogger
from vislogger.util import name_and_iter_to_filename


class PytorchPlotFileLogger(NumpyPlotFileLogger):
    """
    Visual logger, inherits the NumpyPlotLogger and plots/ logs pytorch tensors and variables as files on the local
    file system.
    """

    def __init__(self, *args, **kwargs):
        super(PytorchPlotFileLogger, self).__init__(*args, **kwargs)

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
    @threaded
    def save_image_static(image_dir,
                          tensor,
                          name,
                          n_iter=None,
                          iter_format="{:05d}",
                          prefix=False,
                          save_image_args=None):
        """saves an image"""

        if save_image_args is None: save_image_args = {}

        if n_iter is not None:
            name = name_and_iter_to_filename(name=name,
                                             n_iter=n_iter,
                                             ending=".png",
                                             iter_format=iter_format,
                                             prefix=prefix)

        img_file = os.path.join(image_dir, name)
        tv_save_image(tensor=tensor, filename=img_file, **save_image_args)

    def save_image(self,
                   tensor,
                   name,
                   n_iter=None,
                   iter_format="{:05d}",
                   prefix=False,
                   save_image_args=None):
        """saves an image"""

        if save_image_args is None: save_image_args = {}

        self.save_image_static(image_dir=self.img_dir,
                               tensor=tensor,
                               name=name,
                               n_iter=n_iter,
                               iter_format=iter_format,
                               prefix=prefix,
                               save_image_args=save_image_args)

    @staticmethod
    @threaded
    def save_images_static(image_dir,
                           tensors,
                           n_iter=None,
                           iter_format="{:05d}",
                           prefix=False,
                           save_image_args=None):

        assert isinstance(tensors, dict)
        if save_image_args is None: save_image_args = {}

        for name, tensor in tensors.items():
            PytorchPlotFileLogger.save_image_static(image_dir=image_dir,
                                                    tensor=tensor,
                                                    name=name,
                                                    n_iter=n_iter,
                                                    iter_format=iter_format,
                                                    prefix=prefix,
                                                    save_image_args=save_image_args)

    def save_images(self,
                    tensors,
                    n_iter=None,
                    iter_format="{:05d}",
                    prefix=False,
                    save_image_args=None):

        assert isinstance(tensors, dict)
        if save_image_args is None: save_image_args = {}

        self.save_images_static(image_dir=self.img_dir,
                                tensors=tensors,
                                n_iter=n_iter,
                                iter_format=iter_format,
                                prefix=prefix,
                                save_image_args=save_image_args)

    @staticmethod
    @threaded
    def save_image_grid_static(image_dir,
                               tensor,
                               name,
                               n_iter=None,
                               prefix=False,
                               iter_format="{:05d}",
                               save_image_args=None):

        if n_iter is not None:
            name = name_and_iter_to_filename(name=name,
                                             n_iter=n_iter,
                                             ending=".png",
                                             iter_format=iter_format,
                                             prefix=prefix)
        elif not name.endswith(".png"):
            name += ".png"

        img_file = os.path.join(image_dir, name)

        if save_image_args is None: save_image_args = {}

        tv_save_image(tensor=tensor,
                      filename=img_file,
                      **save_image_args)

    def save_image_grid(self,
                        tensor,
                        name,
                        n_iter=None,
                        prefix=False,
                        iter_format="{:05d}",
                        save_image_args=None):

        if save_image_args is None: save_image_args = {}

        self.save_image_grid_static(image_dir=self.img_dir,
                                    tensor=tensor,
                                    name=name,
                                    n_iter=n_iter,
                                    prefix=prefix,
                                    iter_format=iter_format,
                                    save_image_args=save_image_args)

    def show_image(self,
                   image,
                   name,
                   n_iter=None,
                   iter_format="{:05d}",
                   prefix=False,
                   save_image_args=None,
                   **kwargs):
        self.save_image(tensor=image,
                        name=name,
                        n_iter=n_iter,
                        iter_format=iter_format,
                        save_image_args=save_image_args,
                        prefix=prefix)

    def show_images(self,
                    images,
                    name,
                    n_iter=None,
                    iter_format="{:05d}",
                    prefix=False,
                    save_image_args=None,
                    **kwargs):

        tensors = {}
        for i, img in enumerate(images):
            tensors[name + "_" + str(i)] = img

        self.save_images(tensors=tensors,
                         n_iter=n_iter,
                         iter_format=iter_format,
                         prefix=prefix,
                         save_image_args=save_image_args)

    def show_image_grid(self,
                        images,
                        name,
                        n_iter=None,
                        prefix=False,
                        iter_format="{:05d}",
                        save_image_args=None,
                        **kwargs):

        self.save_image_grid(tensor=images,
                             name=name,
                             n_iter=n_iter,
                             prefix=prefix,
                             iter_format=iter_format,
                             save_image_args=save_image_args)
