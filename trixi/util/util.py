import ast
import importlib
import io
import json
import logging
import math

import numpy as np
import os
import random
import string
import matplotlib.pyplot as plt
import time
import traceback
import warnings
from collections import defaultdict, deque
from hashlib import sha256
from tempfile import gettempdir
from types import FunctionType, ModuleType

import numpy as np
import portalocker
from scipy.misc import imsave

try:
    import torch
except ImportError as e:
    import warnings
    warnings.warn(ImportWarning("Could not import Pytorch related modules:\n%s"
        % e.msg))


    class torch:
        dtype = None


class CustomJSONEncoder(json.JSONEncoder):

    def _encode(self, obj):
        raise NotImplementedError

    def _encode_switch(self, obj):
        if isinstance(obj, list):
            return [self._encode_switch(item) for item in obj]
        elif isinstance(obj, dict):
            return {self._encode_key(key): self._encode_switch(val) for key, val in obj.items()}
        else:
            return self._encode(obj)

    def _encode_key(self, obj):
        return self._encode(obj)

    def encode(self, obj):
        return super(CustomJSONEncoder, self).encode(self._encode_switch(obj))

    def iterencode(self, obj, *args, **kwargs):
        return super(CustomJSONEncoder, self).iterencode(self._encode_switch(obj), *args, **kwargs)


class MultiTypeEncoder(CustomJSONEncoder):

    def _encode_key(self, obj):
        if isinstance(obj, int):
            return "__int__({})".format(obj)
        elif isinstance(obj, float):
            return "__float__({})".format(obj)
        else:
            return self._encode(obj)

    def _encode(self, obj):
        if isinstance(obj, tuple):
            return "__tuple__({})".format(obj)
        elif isinstance(obj, np.integer):
            return "__int__({})".format(obj)
        elif isinstance(obj, np.floating):
            return "__float__({})".format(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return obj


class ModuleMultiTypeEncoder(MultiTypeEncoder):

    def _encode(self, obj, strict=False):
        if type(obj) == type:
            return "__type__({}.{})".format(obj.__module__, obj.__name__)
        elif type(obj) == torch.dtype:
            return "__type__({})".format(str(obj))
        elif isinstance(obj, FunctionType):
            return "__function__({}.{})".format(obj.__module__, obj.__name__)
        elif isinstance(obj, ModuleType):
            return "__module__({})".format(obj.__name__)
        else:
            try:
                return super(ModuleMultiTypeEncoder, self)._encode(obj)
            except Exception as e:
                if strict:
                    raise e
                else:
                    message = "Could not pickle object of type {}\n".format(type(obj))
                    message += traceback.format_exc()
                    warnings.warn(message)
                    return repr(obj)


class CustomJSONDecoder(json.JSONDecoder):

    def _decode(self, obj):
        raise NotImplementedError

    def _decode_switch(self, obj):
        if isinstance(obj, list):
            return [self._decode_switch(item) for item in obj]
        elif isinstance(obj, dict):
            return {self._decode_key(key): self._decode_switch(val) for key, val in obj.items()}
        else:
            return self._decode(obj)

    def _decode_key(self, obj):
        return self._decode(obj)

    def decode(self, obj):
        return self._decode_switch(super(CustomJSONDecoder, self).decode(obj))


class MultiTypeDecoder(CustomJSONDecoder):

    def _decode(self, obj):
        if isinstance(obj, str):
            if obj.startswith("__int__"):
                return int(obj[8:-1])
            elif obj.startswith("__float__"):
                return float(obj[10:-1])
            elif obj.startswith("__tuple__"):
                return tuple(ast.literal_eval(obj[10:-1]))
        return obj


class ModuleMultiTypeDecoder(MultiTypeDecoder):

    def _decode(self, obj):
        if isinstance(obj, str):
            if obj.startswith("__type__"):
                str_ = obj[9:-1]
                module_ = ".".join(str_.split(".")[:-1])
                name_ = str_.split(".")[-1]
                type_ = str_
                try:
                    type_ = getattr(importlib.import_module(module_), name_)
                except:
                    warnings.warn("Could not load {}".format(str_))
                return type_
            elif obj.startswith("__function__"):
                str_ = obj[13:-1]
                module_ = ".".join(str_.split(".")[:-1])
                name_ = str_.split(".")[-1]
                type_ = str_
                try:
                    type_ = getattr(importlib.import_module(module_), name_)
                except:
                    warnings.warn("Could not load {}".format(str_))
                return type_
            elif obj.startswith("__module__"):
                str_ = obj[11:-1]
                type_ = str_
                try:
                    type_ = importlib.import_module(str_)
                except:
                    warnings.warn("Could not load {}".format(str_))
                return type_
        return super(ModuleMultiTypeDecoder, self)._decode(obj)


class StringMultiTypeDecoder(CustomJSONDecoder):

    def _decode(self, obj):
        if isinstance(obj, str):
            if obj.startswith("__int__"):
                return obj[8:-1]
            elif obj.startswith("__float__"):
                return obj[10:-1]
            elif obj.startswith("__tuple__"):
                return obj[10:-1]
            elif obj.startswith("__type__"):
                return obj[9:-1]
            elif obj.startswith("__function__"):
                return obj[13:-1]
            elif obj.startswith("__module__"):
                return obj[11:-1]
        return obj


class Singleton:
    """
    A non-thread-safe helper class to ease implementing singletons.
    This should be used as a decorator -- not a metaclass -- to the
    class that should be a singleton.

    The decorated class can define one `__init__` function that
    takes only the `self` argument. Also, the decorated class cannot be
    inherited from. Other than that, there are no restrictions that apply
    to the decorated class.

    To get the singleton instance, use the `Instance` method. Trying
    to use `__call__` will result in a `TypeError` being raised.

    """

    _instance = None

    def __init__(self, decorated):
        self._decorated = decorated

    def get_instance(self, **kwargs):
        """
        Returns the singleton instance. Upon its first call, it creates a
        new instance of the decorated class and calls its `__init__` method.
        On all subsequent calls, the already created instance is returned.

        """
        if not self._instance:
            self._instance = self._decorated(**kwargs)
            return self._instance
        else:
            return self._instance

    def __call__(self):
        raise TypeError('Singletons must be accessed through `get_instance()`.')
        # return self.get_instance()

    def __instancecheck__(self, inst):
        return isinstance(inst, self._decorated)


def get_image_as_buffered_file(image_array):
    """
    Returns a images as file pointer in a buffer

    Args:
        image_array: (C,W,H) To be returned as a file pointer

    Returns:
        Buffer file-pointer object containing the image file
    """
    buf = io.BytesIO()
    imsave(name=buf, arr=image_array.transpose((1, 2, 0)), format="png")
    buf.seek(0)

    return buf


def figure_to_image(figures, close=True):
    """Render matplotlib figure to numpy format.

    Note that this requires the ``matplotlib`` package.
    (https://tensorboardx.readthedocs.io/en/latest/_modules/tensorboardX/utils.html#figure_to_image)

    Args:
        figure (matplotlib.pyplot.figure) or list of figures: figure or a list of figures
        close (bool): Flag to automatically close the figure

    Returns:
        numpy.array: image in [CHW] order
    """
    try:
        import matplotlib.pyplot as plt
        import matplotlib.backends.backend_agg as plt_backend_agg
    except ModuleNotFoundError:
        print('please install matplotlib')

    def render_to_rgb(figure):
        canvas = plt_backend_agg.FigureCanvasAgg(figure)
        canvas.draw()
        data = np.frombuffer(canvas.buffer_rgba(), dtype=np.uint8)
        w, h = figure.canvas.get_width_height()
        image_hwc = data.reshape([h, w, 4])[:, :, 0:3]
        image_chw = np.moveaxis(image_hwc, source=2, destination=0)
        if close:
            plt.close(figure)
        return image_chw

    if isinstance(figures, list):
        images = [render_to_rgb(figure) for figure in figures]
        return np.stack(images)
    else:
        image = render_to_rgb(figures)
        return image


def savefig_and_close(figure, filename, close=True):
    fig_img = figure_to_image(figure, close=close)
    imsave(filename, np.transpose(fig_img, (1, 2, 0)))


def random_string(length):
    random.seed()
    return "".join(random.choice(string.ascii_letters + string.digits) for _ in range(length))


def create_folder(path):
    """
    Creates a folder if not already exists
    Args:
        :param path: The folder to be created
    Returns
        :return: True if folder was newly created, false if folder already exists
    """

    if not os.path.exists(path):
        os.makedirs(path)
        return True
    else:
        return False


def name_and_iter_to_filename(name, n_iter, ending, iter_format="{:05d}", prefix=False):
    iter_str = iter_format.format(n_iter)
    if prefix:
        name = iter_str + "_" + name + ending
    else:
        name = name + "_" + iter_str + ending

    return name


class SafeDict(dict):
    def __missing__(self, key):
        return "{" + key + "}"


class PyLock(object):
    def __init__(self, name, timeout, check_interval=0.25):
        self._timeout = timeout
        self._check_interval = check_interval

        lock_directory = gettempdir()
        unique_token = sha256(name.encode()).hexdigest()
        self._filepath = os.path.join(lock_directory, 'ilock-' + unique_token + '.lock')

    def __enter__(self):

        current_time = call_time = time.time()
        while call_time + self._timeout > current_time:
            self._lockfile = open(self._filepath, 'w')
            try:
                portalocker.lock(self._lockfile, portalocker.constants.LOCK_NB | portalocker.constants.LOCK_EX)
                return self
            except portalocker.exceptions.LockException:
                pass

            current_time = time.time()
            check_interval = self._check_interval if self._timeout > self._check_interval else self._timeout
            time.sleep(check_interval)

        raise RuntimeError('Timeout was reached')

    def __exit__(self, exc_type, exc_val, exc_tb):
        portalocker.unlock(self._lockfile)
        self._lockfile.close()


class LogDict(dict):
    def __init__(self, file_name, base_dir=None, to_console=False, mode="a"):
        """Initializes a new Dict which can log to a given target file."""

        super(LogDict, self).__init__()

        self.file_name = file_name
        if base_dir is not None:
            self.file_name = os.path.join(base_dir, file_name)

        self.logging_identifier = random_string(15)
        self.logger = logging.getLogger("logdict-" + self.logging_identifier)
        self.logger.setLevel(logging.INFO)
        file_handler_formatter = logging.Formatter('')

        self.file_handler = logging.FileHandler(self.file_name, mode=mode)
        self.file_handler.setFormatter(file_handler_formatter)
        self.logger.addHandler(self.file_handler)
        self.logger.propagate = to_console

    def __setitem__(self, key, item):
        super(LogDict, self).__setitem__(key, item)

    def log_complete_content(self):
        """Logs the current content of the dict to the output file as a whole."""
        self.logger.info(str(self))


class ResultLogDict(LogDict):
    def __init__(self, file_name, base_dir=None, running_mean_length=10, **kwargs):
        """Initializes a new Dict which directly logs value changes to a given target_file."""
        super(ResultLogDict, self).__init__(file_name=file_name, base_dir=base_dir, **kwargs)

        self.is_init = False
        self.running_mean_dict = defaultdict(lambda: deque(maxlen=running_mean_length))

        self.__cntr_dict = defaultdict(float)

        if self.file_handler.mode == "w" or os.stat(self.file_handler.baseFilename).st_size == 0:
            self.print_to_file("[")

        self.is_init = True

    def __setitem__(self, key, item):

        if key == "__cntr_dict":
            raise ValueError("In ResultLogDict you can not add an item with key '__cntr_dict'")

        data = item
        if isinstance(item, dict) and "data" in item and "label" in item and "epoch" in item:

            data = item["data"]
            if "counter" in item and item["counter"] is not None:
                self.__cntr_dict[key] = item["counter"]
            json_dict = {key: ResultElement(data=data, label=item["label"], epoch=item["epoch"],
                                            counter=self.__cntr_dict[key])}
        else:
            json_dict = {key: ResultElement(data=data, counter=self.__cntr_dict[key])}
        self.__cntr_dict[key] += 1
        self.logger.info(json.dumps(json_dict) + ",")

        self.running_mean_dict[key].append(data)

        super(ResultLogDict, self).__setitem__(key, data)

    def print_to_file(self, text):
        self.logger.info(text)

    def load(self, reload_dict):
        for key, item in reload_dict.items():

            if isinstance(item, dict) and "data" in item and "label" in item and "epoch" in item:
                data = item["data"]
                if "counter" in item and item["counter"] is not None:
                    self.__cntr_dict[key] = item["counter"]
            else:
                data = item
            self.__cntr_dict[key] += 1

            super(ResultLogDict, self).__setitem__(key, data)

    def close(self):

        self.file_handler.close()
        # Remove trailing comma, unless we've only written "[".
        # This approach (fixed offset) sometimes fails upon errors and the like,
        # we could alternatively read the whole file,
        # parse to only keep "clean" rows and rewrite.
        with open(self.file_handler.baseFilename, "rb+") as handle:
            if os.stat(self.file_handler.baseFilename).st_size > 2:
                handle.seek(-2, os.SEEK_END)
                handle.truncate()
        with open(self.file_handler.baseFilename, "a") as handle:
            handle.write("\n]")


class ResultElement(dict):
    def __init__(self, data=None, label=None, epoch=None, counter=None):
        super(ResultElement, self).__init__()

        if data is not None:
            if isinstance(data, np.floating):
                data = float(data)
            if isinstance(data, np.integer):
                data = int(data)
            self["data"] = data
        if label is not None:
            self["label"] = label
        if epoch is not None:
            self["epoch"] = epoch
        if counter is not None:
            self["counter"] = counter


def chw_to_hwc(np_array):
    if len(np_array.shape) != 3:
        return np_array
    elif np_array.shape[0] != 1 and np_array.shape[0] != 3:
        return np_array
    elif np_array.shape[2] == 1 or np_array.shape[2] == 3:
        return np_array
    else:
        np_array = np.transpose(np_array, (1, 2, 0))
        return np_array


def np_make_grid(np_array, nrow=8, padding=2,
                 normalize=False, range_=None, scale_each=False, pad_value=0, to_int=False):
    """Make a grid of images.

    Args:
        np_array (numpy array): 4D mini-batch Tensor of shape (B x C x H x W)
            or a list of images all of the same size.
        nrow (int, optional): Number of images displayed in each row of the grid.
            The Final grid size is (B / nrow, nrow). Default is 8.
        padding (int, optional): amount of padding. Default is 2.
        normalize (bool, optional): If True, shift the image to the range (0, 1),
            by subtracting the minimum and dividing by the maximum pixel value.
        range_ (tuple, optional): tuple (min, max) where min and max are numbers,
            then these numbers are used to normalize the image. By default, min and max
            are computed from the tensor.
        scale_each (bool, optional): If True, scale each image in the batch of
            images separately rather than the (min, max) over all images.
        pad_value (float, optional): Value for the padded pixels.
        to_int (bool): Transforms the np array to a unit8 array with min 0 and max 255

    Example:
        See this notebook `here <https://gist.github.com/anonymous/bf16430f7750c023141c562f3e9f2a91>`_

    """
    if not (isinstance(np_array, np.ndarray) or
            (isinstance(np_array, list) and all(isinstance(a, np.ndarray) for a in np_array))):
        raise TypeError('Numpy array or list of tensors expected, got {}'.format(type(np_array)))

    # if list of tensors, convert to a 4D mini-batch Tensor
    if isinstance(np_array, list):
        np_array = np.stack(np_array, axis=0)

    if len(np_array.shape) == 2:  # single image H x W
        np_array = np_array.reshape((1, np_array.shape[0], np_array.shape[1]))
    if len(np_array.shape) == 3:  # single image
        if np_array.shape[0] == 1:  # if single-channel, convert to 3-channel
            np_array = np.concatenate((np_array, np_array, np_array), 0)
        np_array = np_array.reshape((1, np_array.shape[0], np_array.shape[1], np_array.shape[2]))

    if len(np_array.shape) == 3 == 4 and np_array.shape[1] == 1:  # single-channel images
        np_array = np.concatenate((np_array, np_array, np_array), 1)

    if normalize is True:
        np_array = np.copy(np_array)  # avoid modifying tensor in-place
        if range_ is not None:
            assert isinstance(range_, tuple), \
                "range has to be a tuple (min, max) if specified. min and max are numbers"

        def norm_ip(img, min_, max_):
            img = np.clip(img, a_min=min_, a_max=max_)
            img = (img - min_) / (max_ - min_ + 1e-5)
            return img

        def norm_range(t, range__=None):
            if range__ is not None:
                t = norm_ip(t, range__[0], range__[1])
            else:
                t = norm_ip(t, float(t.min()), float(t.max()))
            return t

        if scale_each is True:
            for i in range(np_array.shape[0]):  # loop over mini-batch dimension
                np_array[i] = norm_range(np_array[i], range_)
        else:
            np_array = norm_range(np_array, range_)

    if np_array.shape[0] == 1:
        return np_array.squeeze(0)

    # make the mini-batch of images into a grid
    nmaps = np_array.shape[0]
    xmaps = min(nrow, nmaps)
    ymaps = int(math.ceil(float(nmaps) / xmaps))
    height, width = int(np_array.shape[2] + padding), int(np_array.shape[3] + padding)
    grid = np.zeros((3, height * ymaps + padding, width * xmaps + padding))
    grid += pad_value
    k = 0
    for y in range(ymaps):
        for x in range(xmaps):
            if k >= nmaps:
                break
            grid[:,
            y * height + padding: y * height + padding + height - padding,
            x * width + padding: x * width + padding + width - padding] = np_array[k]
            k = k + 1

    if to_int:
        grid = np.clip(grid * 255, a_min=0, a_max=255)
        grid = grid.astype(np.uint8)

    return grid


def get_tensor_embedding(tensor, method="tsne", n_dims=2, n_neigh=30, **meth_args):
    """
    Return a embedding of a tensor (in a lower dimensional space, e.g. t-SNE)

    Args:
       tensor: Tensor to be embedded
       method: Method used for embedding, options are: tsne, standard, ltsa, hessian, modified, isomap, mds,
       spectral, umap
       n_dims: dimensions to embed the data into
       n_neigh: Neighbour parameter to kind of determin the embedding (see t-SNE for more information)
       **meth_args: Further arguments which can be passed to the embedding method

    Returns:
        The embedded tensor

    """
    from sklearn import manifold
    import umap

    linears = ['standard', 'ltsa', 'hessian', 'modified']
    if method in linears:

        loclin = manifold.LocallyLinearEmbedding(n_neigh, n_dims, method=method, **meth_args)
        emb_data = loclin.fit_transform(tensor)

    elif method == "isomap":
        iso = manifold.Isomap(n_neigh, n_dims, **meth_args)
        emb_data = iso.fit_transform(tensor)

    elif method == "mds":
        mds = manifold.MDS(n_dims, **meth_args)
        emb_data = mds.fit_transform(tensor)

    elif method == "spectral":
        se = manifold.SpectralEmbedding(n_components=n_dims, n_neighbors=n_neigh, **meth_args)
        emb_data = se.fit_transform(tensor)

    elif method == "tsne":
        tsne = manifold.TSNE(n_components=n_dims, perplexity=n_neigh, **meth_args)
        emb_data = tsne.fit_transform(tensor)

    elif method == "umap":
        um = umap.UMAP(n_components=n_dims, n_neighbors=n_neigh, **meth_args)
        emb_data = um.fit_transform(tensor)

    else:
        emb_data = tensor

    return emb_data
