from __future__ import print_function

import datetime
import json
import os
try:
    import cPickle as pickle
except:
    import pickle

import numpy as np

from vislogger import AbstractLogger, FileLogger, NumpyPlotLogger, Config
from vislogger.util import create_folder, MultiTypeEncoder, MultiTypeDecoder


class ExperimentLogger(AbstractLogger):
    """A single class for logging"""

    def __init__(self,
                 experiment_name,
                 base_dir,
                 folder_format="{experiment_name}-{run_number:04d}",
                 **kwargs):

        super(ExperimentLogger, self).__init__(**kwargs)

        self.experiment_name = experiment_name
        self.base_dir = base_dir
        self.folder_format = folder_format

        self.init_time = datetime.datetime.today()

        self.folder_name = self.resolve_format(folder_format)
        self.work_dir = os.path.join(base_dir, self.folder_name)

        self.config_dir = os.path.join(self.work_dir, "config")
        self.log_dir = os.path.join(self.work_dir, "log")
        self.checkpoint_dir = os.path.join(self.work_dir, "checkpoint")
        self.img_dir = os.path.join(self.work_dir, "img")
        self.plot_dir = os.path.join(self.work_dir, "plot")
        self.save_dir = os.path.join(self.work_dir, "save")

        create_folder(self.work_dir)
        create_folder(self.config_dir)
        create_folder(self.log_dir)
        create_folder(self.checkpoint_dir)
        create_folder(self.img_dir)
        create_folder(self.plot_dir)
        create_folder(self.save_dir)

        self.file_logger = FileLogger(self.work_dir)
        self.plot_logger = NumpyPlotLogger(self.img_dir, self.plot_dir)

    def show_image(self, image, name, file_format=".png", **kwargs):
        self.plot_logger.show_image(image, name, file_format=".png", **kwargs)

    def show_barplot(self, array, name, file_format=".png", **kwargs):
        self.plot_logger.show_barplot(array, name, file_format=".png", **kwargs)

    def show_lineplot(self, y_vals, x_vals, name, file_format=".png", **kwargs):
        self.plot_logger.show_lineplot(y_vals, x_vals, name, file_format=".png", **kwargs)

    def show_piechart(self, array, name, file_format=".png", **kwargs):
        self.plot_logger.show_piechart(array, name, file_format=".png", **kwargs)

    def show_scatterplot(self, array, name, file_format=".png", **kwargs):
        self.plot_logger.show_scatterplot(array, name, file_format=".png", **kwargs)

    def show_value(self, value, name=None, logger="default", **kwargs):
        self.plot_logger.show_value(value, name, logger, **kwargs)

    def show_text(self, text, name=None, logger="default", **kwargs):
        self.file_logger.show_text(text, name, logger, **kwargs)

    def save_model(self):
        raise NotImplementedError

    def load_model(self):
        raise NotImplementedError

    def save_config(self, data, name):

        if not name.endswith(".json"):
            name += ".json"
        data.dump(os.path.join(self.config_dir, name))

    def load_config(self, name):

        if not name.endswith(".json"):
            name += ".json"
        c = Config()
        c.load(os.path.join(self.config_dir, name))
        return c

    def save_checkpoint(self):
        raise NotImplementedError

    def load_checkpoint(self):
        raise NotImplementedError

    def save_dict(self, data, path):

        if not path.endswith(".json"):
            path += ".json"
        path = os.path.join(self.save_dir, path)
        create_folder(os.path.dirname(path))
        json.dump(data, open(path, "w"), cls=MultiTypeEncoder)

    def load_dict(self, path):

        if not path.endswith(".json"):
            path += ".json"
        path = os.path.join(self.save_dir, path)
        return json.load(open(path, "r"), cls=MultiTypeDecoder)

    def save_numpy_data(self, data, path):

        if not path.endswith(".npy"):
            path += ".npy"
        path = os.path.join(self.save_dir, path)
        create_folder(os.path.dirname(path))
        np.save(path, data)

    def load_numpy_data(self, path):

        if not path.endswith(".npy"):
            path += ".npy"
        path = os.path.join(self.save_dir, path)
        return np.load(path)

    def save_pickle(self, data, path):

        path = os.path.join(self.save_dir, path)
        create_folder(os.path.dirname(path))
        with open(path, "wb") as out:
            pickle.dump(data, out)

    def load_pickle(self, path):

        path = os.path.join(self.save_dir, path)
        with open(path, "rb") as in_:
            return pickle.load(in_)

    def resolve_format(self, input_):

        if "%" in input_:
            input_ = self.init_time.strftime(input_)

        if "{" not in input_:
            return input_

        run_number = 1
        while os.path.exists(os.path.join(self.base_dir,
                                          input_.format(run_number=run_number,
                                                        **self.__dict__))):
            run_number += 1

        return input_.format(run_number=run_number, **self.__dict__)


