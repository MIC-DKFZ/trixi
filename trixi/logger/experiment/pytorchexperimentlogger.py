from __future__ import print_function

import atexit
import fnmatch
import os
import warnings
from multiprocessing import Process
# import cv2
from PIL import Image

import torch
import numpy as np

from trixi.logger.abstractlogger import threaded
from trixi.logger.experiment import ExperimentLogger
from trixi.logger.file.pytorchplotfilelogger import PytorchPlotFileLogger
from trixi.util import name_and_iter_to_filename
from trixi.util.pytorchutils import update_model, get_vanilla_image_gradient, get_guided_image_gradient, \
    get_smooth_image_gradient


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

    def show_image_grid_heatmap(self, heatmap, background=None, name="heatmap", **kwargs):
        """
        Saves images in the img folder as a image grid

        Args:
            heatmap: The images to be converted to a heatmap
            background: Context of the heatmap (to be underlayed)
            name: file name of the new image file

        """
        self.plot_logger.show_image_grid_heatmap(heatmap=heatmap, background=background, name=name, **kwargs)

    def show_video(self, frame_list=None, name="video", dim="LxHxWxC", scale=1.0, fps=25,
                   extension=".mp4", codec="THEO"):
        """
        Saves video in the img folder. Should be a list of arrays with dimension HxWxC.

        Args:
            frame_list: The list of image tensors/arrays to be saved as a video
            name: Filename of the video
            dim: Dimension of the tensor - should be either LxHxWxC or LxCxHxW
            fps: FPS of the video
            extension: File extension - should be mp4, ogc, avi or webm
        """
        # TODO: trixi browser currently can't show videos, so using GIF instead - work in progress
        self.show_gif(frame_list, name=name, scale=scale, fps=fps)
        """
        tensor = np.array(frame_list)
        assert tensor.ndim == 4, "video should be a 4d tensor"
        assert dim == "LxHxWxC" or  dim == "LxCxHxW", "dimension argument should be LxHxWxC or LxCxHxW"
        if dim == "LxCxHxW":
            tensor = tensor.transpose([0, 2, 3, 1])
        filename = os.path.join(self.img_dir, name + extension)
        fourcc = cv2.VideoWriter_fourcc(*codec)
        writer = cv2.VideoWriter(filename, fourcc, fps, (tensor.shape[2], tensor.shape[1]))
        assert writer.isOpened(), "video writer could not be opened"
        for i in range(tensor.shape[0]):
            writer.write(tensor[i, :, :, :])
        writer.release()
        writer = None
        """

    def show_gif(self, frame_list=None, name="frames", scale=1.0, fps=25):
        """
        Saves gif in the img folder. Should be a list of arrays with dimension HxWxC.

        Args:
            frame_list: The list of image tensors/arrays to be saved as a gif
            name: Filename of the gif
            scale: Scaling factor of the individual frames
            fps: FPS of the gif
        """
        w, h = Image.fromarray(np.uint8(frame_list[0])).size
        image_list = []
        for i in range(len(frame_list)):
            image_list.append(Image.fromarray(np.uint8(frame_list[i])).resize((w*int(scale), h*int(scale))))
        filename = os.path.join(self.img_dir, name + ".gif")
        image_list[0].save(filename, save_all=True, append_images=image_list[1:], duration=int(1e3/fps), loop=0)

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

    @staticmethod
    def get_roc_curve(tensor, labels, reduce_to_n_samples=None, use_sub_process=False, results_fn=lambda
            x, *y, **z: None):
        """
        Displays a roc curve given a tensor with scores and the coresponding labels

        Args:
            tensor: Tensor with scores (e.g class probability )
            labels: Labels of the samples to which the scores match
            reduce_to_n_samples: Reduce/ downsample to to n samples for fewer data points
            use_sub_process: Use a sub process to do the processing, if true nothing is returned
            results_fn: function which is called with the results/ return values. Expected f(tpr, fpr)

        """
        from sklearn import metrics

        def __get_roc_curve(tensor, labels, reduce_to_n_samples=None, results_fn=lambda x, *y, **z: None):

            if not isinstance(labels, list):
                labels = labels.flatten()
            if not isinstance(tensor, list):
                tensor = tensor.flatten()

            fpr, tpr, thresholds = metrics.roc_curve(labels, tensor)
            if reduce_to_n_samples is not None:
                fpr = [np.mean(x) for x in np.array_split(fpr, reduce_to_n_samples)]
                tpr = [np.mean(x) for x in np.array_split(tpr, reduce_to_n_samples)]
            results_fn(tpr, fpr)

            return tpr, fpr
            # self.show_lineplot(tpr, fpr, name=name, opts={"fillarea": True, "webgl": True})
            # self.add_to_graph(x_vals=np.arange(0, 1.1, 0.1), y_vals=np.arange(0, 1.1, 0.1), name=name, append=True)

        if use_sub_process:
            p = Process(target=__get_roc_curve, kwargs=dict(tensor=tensor,
                                                            labels=labels,
                                                            reduce_to_n_samples=reduce_to_n_samples,
                                                            results_fn=results_fn
                                                            ))
            atexit.register(p.terminate)
            p.start()
        else:
            try:
                return __get_roc_curve(tensor=tensor,
                                       labels=labels,
                                       reduce_to_n_samples=reduce_to_n_samples,
                                       results_fn=results_fn
                                       )
            except Exception as e:
                warnings.warn("Sth went wrong with calculating the roc curve")

    @staticmethod
    def get_pr_curve(tensor, labels, reduce_to_n_samples=None, use_sub_process=False,
                     results_fn=lambda x, *y, **z: None):
        """
        Displays a precision recall curve given a tensor with scores and the coresponding labels

        Args:
            tensor: Tensor with scores (e.g class probability )
            labels: Labels of the samples to which the scores match
            reduce_to_n_samples: Reduce/ downsample to to n samples for fewer data points
            use_sub_process: Use a sub process to do the processing, if true nothing is returned
            results_fn: function which is called with the results/ return values. Expected f(precision, recall)

        """
        from sklearn import metrics

        def __get_pr_curve(tensor, labels, reduce_to_n_samples=None, results_fn=lambda x, *y, **z: None):

            if not isinstance(labels, list):
                labels = labels.flatten()
            if not isinstance(tensor, list):
                tensor = tensor.flatten()

            precision, recall, thresholds = metrics.precision_recall_curve(labels, tensor)
            if reduce_to_n_samples is not None:
                precision = [np.mean(x) for x in np.array_split(precision, reduce_to_n_samples)]
                recall = [np.mean(x) for x in np.array_split(recall, reduce_to_n_samples)]
            results_fn(precision, recall)

            return precision, recall
            # self.show_lineplot(precision, recall, name=name, opts={"fillarea": True, "webgl": True})
            # self.add_to_graph(x_vals=np.arange(0, 1.1, 0.1), y_vals=np.arange(0, 1.1, 0.1), name=name, append=True)

        if use_sub_process:
            p = Process(target=__get_pr_curve, kwargs=dict(tensor=tensor,
                                                           labels=labels,
                                                           reduce_to_n_samples=reduce_to_n_samples,
                                                           results_fn=results_fn
                                                           ))
            atexit.register(p.terminate)
            p.start()
        else:
            try:
                return __get_pr_curve(tensor=tensor,
                                      labels=labels,
                                      reduce_to_n_samples=reduce_to_n_samples,
                                      results_fn=results_fn
                                      )
            except Exception as e:
                warnings.warn("Sth went wrong with calculating the pr curve")

    @staticmethod
    def get_classification_metrics(tensor, labels, name="", metric=("roc-auc", "pr-score"), use_sub_process=False,
                                   tag_name=None, results_fn=lambda x, *y, **z: None):
        """
        Displays some classification metrics as line plots in a graph (similar to show value (also uses show value
        for the caluclated values))

        Args:
            tensor: Tensor with scores (e.g class probability )
            labels: Labels of the samples to which the scores match
            name: The name of the window
            metric: List of metrics to calculate. Options are: roc-auc, pr-auc, pr-score, mcc, f1
            reduce_to_n_samples: Reduce/ downsample to to n samples for fewer data points
            tag_name: Name for the tag, if no given use name
            use_sub_process: Use a sub process to do the processing, if true nothing is returned
            results_fn: function which is called with the results/ return values. Expected f(val, name, tag)

        Returns:

        """

        from sklearn import metrics

        def __get_classification_metrics(tensor, labels, name="", metric=("roc-auc", "pr-score"),
                                         tag_name=None, results_fn=lambda x, *y, **z: None):

            vals = []
            tags = []

            if not isinstance(labels, list):
                labels = labels.flatten()
            if not isinstance(tensor, list):
                tensor = tensor.flatten()

            if "roc-auc" in metric:
                roc_auc = metrics.roc_auc_score(labels, tensor)
                vals.append(roc_auc)
                tags.append("roc-auc")
            if "pr-auc" in metric:
                precision, recall, thresholds = metrics.precision_recall_curve(labels, tensor)
                pr_auc = metrics.auc(recall, precision)
                vals.append(pr_auc)
                tags.append("pr-auc")
            if "pr-score" in metric:
                pr_score = metrics.average_precision_score(labels, tensor)
                vals.append(pr_score)
                tags.append("pr-score")
            if "mcc" in metric:
                mcc_score = metrics.matthews_corrcoef(labels, tensor)
                vals.append(mcc_score)
                tags.append("mcc")
            if "f1" in metric:
                f1_score = metrics.f1_score(labels, tensor)
                vals.append(f1_score)
                tags.append("f1")

            for val, tag in zip(vals, tags):
                results_fn(val, name=tag + "-" + name, tag=tag_name)

            return vals, tags

        if use_sub_process:
            p = Process(target=__get_classification_metrics, kwargs=dict(tensor=tensor,
                                                                         labels=labels,
                                                                         name=name,
                                                                         metric=metric,
                                                                         tag_name=tag_name,
                                                                         results_fn=results_fn
                                                                         ))
            atexit.register(p.terminate)
            p.start()
        else:
            try:
                return __get_classification_metrics(tensor=tensor,
                                                    labels=labels,
                                                    name=name,
                                                    metric=metric,
                                                    tag_name=tag_name,
                                                    results_fn=results_fn
                                                    )

            except Exception as e:
                warnings.warn("Sth went wrong with calculating the classification metrics")

    @staticmethod
    def get_input_gradient(model, inpt, err_fn, grad_type="vanilla", n_runs=20, eps=0.1,
                           abs=False, results_fn=lambda x, *y, **z: None):
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
            results_fn: function which is called with the results/ return values. Expected f(grads)

        """
        model.zero_grad()

        if grad_type == "vanilla":
            grad = get_vanilla_image_gradient(model, inpt, err_fn, abs)
        elif grad_type == "guided":
            grad = get_guided_image_gradient(model, inpt, err_fn, abs)
        elif grad_type == "smooth-vanilla":
            grad = get_smooth_image_gradient(model, inpt, err_fn, abs, n_runs, eps, grad_type="vanilla")
        elif grad_type == "smooth-guided":
            grad = get_smooth_image_gradient(model, inpt, err_fn, abs, n_runs, eps, grad_type="guided")
        else:
            warnings.warn("This grad_type is not implemented yet")
            grad = torch.zeros_like(inpt)
        model.zero_grad()

        results_fn(grad)

        return grad

    def show_image_gradient(self, name, *args, **kwargs):
        """
        Given a model creates calculates the error and backpropagates it to the image and saves it.

        Args:
            name: Name of the file
            model: The model to be evaluated
            inpt: Input to the model
            err_fn: The error function the evaluate the output of the model on
            grad_type: Gradient calculation method, currently supports (vanilla, vanilla-smooth, guided,
            guided-smooth) ( the guided backprob can lead to segfaults -.-)
            n_runs: Number of runs for the smooth variants
            eps: noise scaling to be applied on the input image (noise is drawn from N(0,1))
            abs (bool): Flag, if the gradient should be a absolute value


        """
        grad = self.get_input_gradient(*args, **kwargs)
        self.show_image_grid(grad, name)
