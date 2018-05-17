from __future__ import print_function

import atexit
import fnmatch
import os

import torch

from trixi.logger.abstractlogger import threaded
from trixi.logger.experiment import ExperimentLogger
from trixi.logger.file.pytorchplotfilelogger import PytorchPlotFileLogger
from trixi.util import name_and_iter_to_filename
from trixi.util.pytorchutils import update_model


class PytorchExperimentLogger(ExperimentLogger):
    """
    A single class for logging your pytorch experiments to file.
    Extends the ExperimentLogger also also creates a experiment folder with a file structure:

    The folder structure is :
        base_dir/
            new_experiment_folder/
                checkpoint/
                config/
                img/
                log/
                plot/
                result/
                save/


    """

    def __init__(self, *args, **kwargs):
        """Initializes the PytorchExperimentLogger and parses the arguments to the ExperimentLogger"""

        super(PytorchExperimentLogger, self).__init__(*args, **kwargs)
        self.plot_logger = PytorchPlotFileLogger(self.img_dir, self.plot_dir)

    def show_images(self, images, name, **kwargs):
        """
        Saves images in the img folder

        Args:
            images: The images to be saved
            name: file name of the new image file

        """
        self.plot_logger.show_images(images, name, **kwargs)

    def show_image_grid(self, image, name, **kwargs):
        """
        Saves images in the img folder as a image grid

        Args:
            images: The images to be saved
            name: file name of the new image file

        """
        self.plot_logger.show_image_grid(image, name, **kwargs)

    def show_image_gradient(self, *args, **kwargs):
        """
        Given a model creates calculates the error and backpropagates it to the image and saves it.

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
        self.plot_logger.show_image_gradient(*args, **kwargs)

    @staticmethod
    @threaded
    def save_model_static(model, model_dir, name):
        """
        Saves a pytorch model in a given directory (using pytorch)

        Args:
            model: The model to be stored
            model_dir: The directory in which the model file should be written
            name: The file name of the model file

        """

        model_file = os.path.join(model_dir, name)
        torch.save(model.state_dict(), model_file)

    def save_model(self, model, name, n_iter=None, iter_format="{:05d}", prefix=False):
        """
        Saves a pytorch model in the model directory of the experiment folder

        Args:
            model: The model to be stored
            name: The file name of the model file
            n_iter: The iteration number, formatted with the iter_format and added to the model name (if not None)
            iter_format: The format string, which indicates how n_iter will be formated as a string
            prefix: If True, the formated n_iter will be appended as a prefix, otherwise as a suffix

        """

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
        """
        Loads a pytorch model from a given directory (using pytorch)


        Args:
            model: The model to be loaded (whose parameters should be restored)
            model_file: The file from which the model parameters should be loaded
            exclude_layers: List of layer names which should be excluded from restoring
            warnings (bool): Flag which indicates if method should warn if not everything went perfectly

        """

        if os.path.exists(model_file):

            pretrained_dict = torch.load(model_file, map_location=lambda storage, loc: storage)
            update_model(model, pretrained_dict, exclude_layers, warnings)

        else:

            raise IOError("Model file does not exist!")

    def load_model(self, model, name, exclude_layers=(), warnings=True):
        """
        Loads a pytorch model from the model directory of the experiment folder


        Args:
            model: The model to be loaded (whose parameters should be restored)
            name: The file name of the model file
            exclude_layers: List of layer names which should be excluded from restoring
            warnings: Flag which indicates if method should warn if not everything went perfectlys


        """

        if not name.endswith(".pth"):
            name += ".pth"

        self.load_model_static(model=model,
                               model_file=os.path.join(self.checkpoint_dir, name),
                               exclude_layers=exclude_layers,
                               warnings=warnings)

    @staticmethod
    @threaded
    def save_checkpoint_static(checkpoint_dir, name, move_to_cpu=False, **kwargs):
        """
        Saves a checkpoint/dict in a given directory (using pytorch)

        Args:
            checkpoint_dir: The directory in which the checkpoint file should be written
            name: The file name of the checkpoint file
            move_to_cpu (bool): Flag, if all pytorch tensors should be moved to cpu before storing
            **kwargs: dict which is actually saved

        """
        for key, value in kwargs.items():
            if isinstance(value, torch.nn.Module) or isinstance(value, torch.optim.Optimizer):
                kwargs[key] = value.state_dict()

        checkpoint_file = os.path.join(checkpoint_dir, name)

        def to_cpu(obj):
            if hasattr(obj, "cpu"):
                return obj.cpu()
            elif isinstance(obj, dict):
                return {key: to_cpu(val) for key, val in obj.items()}
            else:
                return obj

        if move_to_cpu:
            torch.save(to_cpu(kwargs), checkpoint_file)
        else:
            torch.save(kwargs, checkpoint_file)

    def save_checkpoint(self, name, n_iter=None, iter_format="{:05d}", prefix=False, **kwargs):
        """
        Saves a checkpoint in the checkpoint directory of the experiment folder

        Args:
            name: The file name of the checkpoint file
            n_iter: The iteration number, formatted with the iter_format and added to the checkpoint name (if not None)
            iter_format: The format string, which indicates how n_iter will be formated as a string
            prefix: If True, the formated n_iter will be appended as a prefix, otherwise as a suffix
            **kwargs:  dict which is actually saved (key=name, value=variable to be stored)

        """

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
        """
        Loads a checkpoint/dict in a given directory (using pytorch)

        Args:
            checkpoint_file: The checkpoint from which the checkpoint/dict should be loaded
            exclude_layer_dict: A dict with key 'model_name' and a list of all layers of 'model_name' which should
            not be restored
            warnings: Flag which indicates if method should warn if not everything went perfectlys
            **kwargs: dict which is actually loaded (key=name (used to save the checkpoint) , value=variable to be
            loaded/ overwritten)

        Returns: The kwargs dict with the loaded/ overwritten values

        """

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
        """
        Loads a checkpoint from the checkpoint directory of the experiment folder

        Args:
            name: The name of the checkpoint file
            exclude_layer_dict: A dict with key 'model_name' and a list of all layers of 'model_name' which should
            not be restored
            warnings: Flag which indicates if method should warn if not everything went perfectlys
            **kwargs: dict which is actually loaded (key=name (used to save the checkpoint) , value=variable to be
            loaded/ overwritten)

        Returns: The kwargs dict with the loaded/ overwritten values

        """

        if not name.endswith(".pth.tar"):
            name += ".pth.tar"

        checkpoint_file = os.path.join(self.checkpoint_dir, name)
        return self.load_checkpoint_static(checkpoint_file=checkpoint_file,
                                           exclude_layer_dict=exclude_layer_dict,
                                           warnings=warnings,
                                           **kwargs)

    def save_at_exit(self, name="checkpoint_end", **kwargs):
        """
        Saves a dict as checkpoint if the program exits (not garanteed to work 100%)

        Args:
            name: Name of the checkpoint file
            **kwargs: dict which is actually saved (key=name, value=variable to be stored)

        """

        if not name.endswith(".pth.tar"):
            name += ".pth.tar"

        def save_fnc():
            self.save_checkpoint(name, **kwargs)
            print("Checkpoint saved securely... =)")

        atexit.register(save_fnc)

    def get_save_checkpoint_fn(self, name="checkpoint", **kwargs):
        """
        A function which returns a function which takes n_iter as arguments and saves the current values of the
        variables given as kwargs as a checkpoint file.


        Args:
            name: Base-name of the checkpoint file
            **kwargs:  dict which is actually saved, when the returned function is called

        Returns: Function which takes n_iter as arguments and saves a checkpoint file
        """

        def save_fnc(n_iter, iter_format="{:05d}", prefix=False):
            self.save_checkpoint(name=name,
                                 n_iter=n_iter,
                                 iter_format=iter_format,
                                 prefix=prefix,
                                 **kwargs)
        return save_fnc

    @staticmethod
    def load_last_checkpoint_static(dir_, name=None, **kwargs):
        """
        Loads the (alphabetically) last checkpoint file in a given directory

        Args:
            dir_: The directory to look for the (alphabetically) last checkpoint
            name: String pattern which indicates the files to look form
            **kwargs: dict which is actually loaded (key=name (used to save the checkpoint) , value=variable to be
            loaded/ overwritten)

        Returns:  The kwargs dict with the loaded/ overwritten values

        """

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
        """
                Loads the (alphabetically) last checkpoint file in the checkpoint directory in the experiment folder

                Args:
                    **kwargs: dict which is actually loaded (key=name (used to save the checkpoint) , value=variable to be
                    loaded/ overwritten)

                Returns:  The kwargs dict with the loaded/ overwritten values

                """
        return self.load_last_checkpoint_static(self.checkpoint_dir, **kwargs)

    def print(self, *args):
        """
        Prints the given arguments using the text logger print function

        Args:
            *args: Things to be printed

        """
        self.text_logger.print(*args)
