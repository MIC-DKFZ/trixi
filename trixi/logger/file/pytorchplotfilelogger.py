import os
import warnings

import torch
from torch.autograd import Variable
from torchvision.utils import save_image as tv_save_image

from trixi.logger.abstractlogger import threaded
from trixi.logger.file.numpyplotfilelogger import NumpyPlotFileLogger
from trixi.util import name_and_iter_to_filename
from trixi.util.pytorchutils import get_guided_image_gradient, get_smooth_image_gradient, get_vanilla_image_gradient


class PytorchPlotFileLogger(NumpyPlotFileLogger):
    """
    Visual logger, inherits the NumpyPlotLogger and plots/ logs pytorch tensors and variables as files on the local
    file system.
    """

    def __init__(self, *args, **kwargs):
        """
        Initializes a PytorchPlotFileLogger to plot images, plots, ... into an image and plot directory

        Args:
            img_dir: The directory to store images in
            plot_dir: The directory to store plots in
        """
        super(PytorchPlotFileLogger, self).__init__(*args, **kwargs)

    def process_params(self, f, *args, **kwargs):
        """
        Inherited "decorator": convert Pytorch variables and Tensors to numpy arrays
        """

        ### convert args
        args = (a.detach().cpu().numpy() if torch.is_tensor(a) else a for a in args)
        #args = (a.data.cpu().numpy() if isinstance(a, Variable) else a for a in args)

        ### convert kwargs
        for key, data in kwargs.items():
            if isinstance(data, Variable):
                kwargs[key] = data.data.cpu().numpy()
            elif torch.is_tensor(data):
                kwargs[key] = data.cpu().numpy()

        return f(self, *args, **kwargs)

    @staticmethod
    @threaded
    def save_image_static(image_dir, tensor, name, n_iter=None, iter_format="{:05d}", prefix=False, image_args=None):
        """
        Saves an image tensor in an image directory

        Args:
            image_dir: Directory to save the image in
            tensor: Tensor containing the image
            name: file-name of the image file
            n_iter: The iteration number, formatted with the iter_format and added to the model name (if not None)
            iter_format: The format string, which indicates how n_iter will be formated as a string
            prefix: If True, the formated n_iter will be appended as a prefix, otherwise as a suffix
            image_args: Arguments for the tensorvision save image method
        """

        if image_args is None:
            image_args = {}

        if n_iter is not None:
            name = name_and_iter_to_filename(name=name, n_iter=n_iter, ending=".png", iter_format=iter_format,
                                             prefix=prefix)

        img_file = os.path.join(image_dir, name)
        os.makedirs(os.path.dirname(img_file), exist_ok=True)
        tv_save_image(tensor=tensor, filename=img_file, **image_args)

    def save_image(self, tensor, name, n_iter=None, iter_format="{:05d}", prefix=False, image_args=None):
        """
        Saves an image into the image directory of the PytorchPlotFileLogger

        Args:
            tensor: Tensor containing the image
            name: file-name of the image file
            n_iter: The iteration number, formatted with the iter_format and added to the model name (if not None)
            iter_format: The format string, which indicates how n_iter will be formated as a string
            prefix: If True, the formated n_iter will be appended as a prefix, otherwise as a suffix
            image_args: Arguments for the tensorvision save image method

        """

        if image_args is None:
            image_args = {}

        self.save_image_static(image_dir=self.img_dir, tensor=tensor, name=name, n_iter=n_iter, iter_format=iter_format,
                               prefix=prefix, image_args=image_args)

    @staticmethod
    @threaded
    def save_images_static(image_dir, tensors, n_iter=None, iter_format="{:05d}", prefix=False, image_args=None):
        """
        Saves an image tensors in an image directory

        Args:
            image_dir: Directory to save the image in
            tensors: A dict with file-name-> tensor to plot as image
            n_iter: The iteration number, formatted with the iter_format and added to the model name (if not None)
            iter_format: The format string, which indicates how n_iter will be formated as a string
            prefix: If True, the formated n_iter will be appended as a prefix, otherwise as a suffix
            image_args: Arguments for the tensorvision save image method
        """

        assert isinstance(tensors, dict)
        if image_args is None:
            image_args = {}

        for name, tensor in tensors.items():
            PytorchPlotFileLogger.save_image_static(image_dir=image_dir, tensor=tensor, name=name, n_iter=n_iter,
                                                    iter_format=iter_format, prefix=prefix, image_args=image_args)

    def save_images(self, tensors, n_iter=None, iter_format="{:05d}", prefix=False, image_args=None):
        """
        Saves an image tensors into the image directory of the PytorchPlotFileLogger

        Args:
            tensors: A dict with file-name-> tensor to plot as image
            n_iter: The iteration number, formatted with the iter_format and added to the model name (if not None)
            iter_format: The format string, which indicates how n_iter will be formated as a string
            prefix: If True, the formated n_iter will be appended as a prefix, otherwise as a suffix
            image_args: Arguments for the tensorvision save image method
        """

        assert isinstance(tensors, dict)
        if image_args is None:
            image_args = {}

        self.save_images_static(image_dir=self.img_dir, tensors=tensors, n_iter=n_iter, iter_format=iter_format,
                                prefix=prefix, image_args=image_args)

    @staticmethod
    @threaded
    def save_image_grid_static(image_dir, tensor, name, n_iter=None, prefix=False, iter_format="{:05d}",
                               image_args=None):
        """
        Saves images of a 4d- tensor (N, C, H, W) as a image grid into an image file in a given directory

        Args:
            image_dir: Directory to save the image in
            tensor: 4d- tensor (N, C, H, W)
            name: file-name of the image file
            n_iter: The iteration number, formatted with the iter_format and added to the model name (if not None)
            iter_format: The format string, which indicates how n_iter will be formated as a string
            prefix: If True, the formated n_iter will be appended as a prefix, otherwise as a suffix
            image_args: Arguments for the tensorvision save image method

        """

        if not (tensor.size(1) == 1 or tensor.size(1) == 3):
            warnings.warn("The 1. dimension (channel) has to be either 1 (gray) or 3 (rgb), taking the first "
                          "dimension now !!!")
            tensor = tensor[:, 0:1, ]

        if n_iter is not None:
            name = name_and_iter_to_filename(name=name, n_iter=n_iter, ending=".png", iter_format=iter_format,
                                             prefix=prefix)
        elif not name.endswith(".png"):
            name += ".png"

        img_file = os.path.join(image_dir, name)

        if image_args is None:
            image_args = {}

        os.makedirs(os.path.dirname(img_file), exist_ok=True)

        tv_save_image(tensor=tensor, filename=img_file, **image_args)

    def save_image_grid(self, tensor, name, n_iter=None, prefix=False, iter_format="{:05d}", image_args=None):
        """
        Saves images of a 4d- tensor (N, C, H, W) as a image grid into an image file in the image directory of the
        PytorchPlotFileLogger

        Args:
            tensor: 4d- tensor (N, C, H, W)
            name: file-name of the image file
            n_iter: The iteration number, formatted with the iter_format and added to the model name (if not None)
            iter_format: The format string, which indicates how n_iter will be formated as a string
            prefix: If True, the formated n_iter will be appended as a prefix, otherwise as a suffix
            image_args: Arguments for the tensorvision save image method

        """

        if image_args is None:
            image_args = {}

        self.save_image_grid_static(image_dir=self.img_dir, tensor=tensor, name=name, n_iter=n_iter, prefix=prefix,
                                    iter_format=iter_format, image_args=image_args)

    def show_image(self, image, name, n_iter=None, iter_format="{:05d}", prefix=False, image_args=None, **kwargs):
        """
        Calls the save image method (for abstract logger combatibility)

        Args:
            image: Tensor containing the image
            name: file-name of the image file
            n_iter: The iteration number, formatted with the iter_format and added to the model name (if not None)
            iter_format: The format string, which indicates how n_iter will be formated as a string
            prefix: If True, the formated n_iter will be appended as a prefix, otherwise as a suffix
            image_args: Arguments for the tensorvision save image method


        """
        self.save_image(tensor=image, name=name, n_iter=n_iter, iter_format=iter_format, image_args=image_args,
                        prefix=prefix)

    def show_images(self, images, name, n_iter=None, iter_format="{:05d}", prefix=False, image_args=None, **kwargs):
        """
        Calls the save images method (for abstract logger combatibility)

        Args:
            images: List of Tensors
            name: List of file names (corresponding to the images list)
            n_iter: The iteration number, formatted with the iter_format and added to the model name (if not None)
            iter_format: The format string, which indicates how n_iter will be formated as a string
            prefix: If True, the formated n_iter will be appended as a prefix, otherwise as a suffix
            image_args: Arguments for the tensorvision save image method

        """
        tensors = {}
        for i, img in enumerate(images):
            tensors[name + "_" + str(i)] = img

        self.save_images(tensors=tensors, n_iter=n_iter, iter_format=iter_format, prefix=prefix, image_args=image_args)

    def show_image_grid(self, images, name, n_iter=None, prefix=False, iter_format="{:05d}", image_args=None,
                        **kwargs):
        """
        Calls the save image grid method (for abstract logger combatibility)

        Args:
            images: 4d- tensor (N, C, H, W)
            name: file-name of the image file
            n_iter: The iteration number, formatted with the iter_format and added to the model name (if not None)
            iter_format: The format string, which indicates how n_iter will be formated as a string
            prefix: If True, the formated n_iter will be appended as a prefix, otherwise as a suffix
            image_args: Arguments for the tensorvision save image method


        """

        self.save_image_grid(tensor=images, name=name, n_iter=n_iter, prefix=prefix, iter_format=iter_format,
                             image_args=image_args)

    def show_image_gradient(self, model, inpt, err_fn, grad_type="vanilla", n_runs=20, eps=0.1, abs=False,
                            **image_grid_params):
        """
        Given a model creates calculates the error and backpropagates it to the image and saves it (saliency map).


        Args:
            model: The model to be evaluated
            inpt: Input to the model
            err_fn: The error function the evaluate the output of the model on
            grad_type: Gradient calculation method, currently supports (vanilla, vanilla-smooth, guided,
            guided-smooth) ( the guided backprob can lead to segfaults -.-)
            n_runs: Number of runs for the smooth variants
            eps: noise scaling to be applied on the input image (noise is drawn from N(0,1))
            abs (bool): Flag, if the gradient should be a absolute value
            **image_grid_params: Params for make image grid.

        """

        if grad_type == "vanilla":
            grad = get_vanilla_image_gradient(model, inpt, err_fn, abs)
        elif grad_type == "guided":
            grad = get_guided_image_gradient(model, inpt, err_fn, abs)
        elif grad_type == "smooth-vanilla":
            grad = get_smooth_image_gradient(model, inpt, err_fn, n_runs, eps, grad_type="vanilla")
        elif grad_type == "smooth-guided":
            grad = get_smooth_image_gradient(model, inpt, err_fn, n_runs, eps, grad_type="guided")
        else:
            warnings.warn("This grad_type is not implemented yet")
            grad = torch.zeros_like(inpt)

        self.show_image_grid(grad, **image_grid_params)
