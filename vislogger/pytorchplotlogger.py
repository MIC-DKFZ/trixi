import os

import torch
from torch.autograd import Variable
from torchvision.utils import save_image as tv_save_image

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
    def save_image_static(image_dir, tensor, name, n_iter=None, iter_format="{:05d}", prefix=False, normalize=True):
        """saves an image"""

        if n_iter is not None:
            name = name_and_iter_to_filename(name=name,
                                             n_iter=n_iter,
                                             ending=".png",
                                             iter_format=iter_format,
                                             prefix=prefix)

        img_file = os.path.join(image_dir, name)
        tv_save_image(tensor=tensor, filename=img_file, normalize=normalize)

    def save_image(self, tensor, name, n_iter=None, iter_format="{:05d}", prefix=False, normalize=True):
        """saves an image"""

        PytorchPlotLogger.save_image_static(image_dir=self.image_dir,
                                            tensor=tensor,
                                            name=name,
                                            n_iter=n_iter,
                                            iter_format=iter_format,
                                            prefix=prefix,
                                            normalize=normalize)

    @staticmethod
    def save_images_static(image_dir, tensors, n_iter=None, iter_format="{:05d}", prefix=False, normalize=True):

        assert isinstance(tensors, dict)

        for name, tensor in tensors.items():
            PytorchPlotLogger.save_image_static(image_dir=image_dir,
                                                tensor=tensor,
                                                name=name,
                                                n_iter=n_iter,
                                                iter_format=iter_format,
                                                prefix=prefix,
                                                normalize=normalize)

    def save_images(self, tensors, n_iter=None, iter_format="{:05d}", prefix=False, normalize=True):

        assert isinstance(tensors, dict)
        PytorchPlotLogger.save_images_static(image_dir=self.image_dir,
                                             tensors=tensors,
                                             n_iter=n_iter,
                                             iter_format=iter_format,
                                             prefix=prefix,
                                             normalize=normalize)

    @staticmethod
    def save_image_grid_static(image_dir, tensor, name, n_iter=None, prefix=False, iter_format="{:05d}", nrow=8,
                               padding=2, normalize=False, range_=None, scale_each=False, pad_value=0):

        if n_iter is not None:
            name = name_and_iter_to_filename(name=name,
                                             n_iter=n_iter,
                                             ending=".png",
                                             iter_format=iter_format,
                                             prefix=prefix)
        elif not name.endswith(".png"):
            name += ".png"

        img_file = os.path.join(image_dir, name)

        tv_save_image(tensor=tensor,
                      filename=img_file,
                      normalize=normalize,
                      nrow=nrow,
                      padding=padding,
                      range=range_,
                      scale_each=scale_each,
                      pad_value=pad_value)

    def save_image_grid(self, tensor, name, n_iter=None, prefix=False, iter_format="{:05d}", nrow=8, padding=2,
                        normalize=False, range_=None, scale_each=False, pad_value=0):

        PytorchPlotLogger.save_image_grid_static(image_dir=self.image_dir,
                                                 tensor=tensor,
                                                 name=name,
                                                 n_iter=n_iter,
                                                 prefix=prefix,
                                                 iter_format=iter_format,
                                                 nrow=nrow,
                                                 padding=padding,
                                                 normalize=normalize,
                                                 range_=range_,
                                                 scale_each=scale_each,
                                                 pad_value=pad_value)

    def show_image(self, image, name, n_iter=None, iter_format="{:05d}", prefix=False, **kwargs):
        self.save_image(tensor=image, name=name, n_iter=n_iter, iter_format=iter_format, prefix=prefix)

    def show_images(self, images, name, n_iter=None, iter_format="{:05d}", prefix=False, **kwargs):

        tensors = {}
        for i, img in enumerate(images):
            tensors[name + str(i)] = img

        self.save_images(tensors=tensors, n_iter=n_iter, iter_format=iter_format, prefix=prefix)

    def show_image_grid(self, images, name, n_iter=None, prefix=False, iter_format="{:05d}", nrow=8, padding=2,
                        normalize=False, range_=None, scale_each=False, pad_value=0, **kwargs):

        self.save_image_grid(tensor=images,
                             name=name,
                             n_iter=n_iter,
                             prefix=prefix,
                             iter_format=iter_format,
                             nrow=nrow,
                             padding=padding,
                             normalize=normalize,
                             range_=range_,
                             scale_each=scale_each,
                             pad_value=pad_value)


