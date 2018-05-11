from __future__ import print_function

import datetime
import json
import os
import re
import shutil
import warnings

try:
    import cPickle as pickle
except ImportError:
    import pickle

import numpy as np

from vislogger.logger.abstractlogger import AbstractLogger
from vislogger.logger.file.textlogger import TextLogger
from vislogger.logger.file.numpyplotfilelogger import NumpyPlotFileLogger
from vislogger.util import create_folder, MultiTypeEncoder, MultiTypeDecoder, Config


REPLACEMENTS = [("%Y", 4), ("%m", 2), ("%d", 2), ("%H", 2), ("%M", 2), ("%S", 2),
                ("%w", 1), ("%y", 2), ("%I", 2), ("%f", 6), ("%j", 3), ("%U", 2),
                ("%W", 2)]


class ExperimentLogger(AbstractLogger):
    """A single class for logging"""

    def __init__(self,
                 experiment_name,
                 base_dir,
                 folder_format="%Y%m%d-%H%M%S_{experiment_name}",
                 resume=False,
                 text_logger_args=None,
                 plot_logger_args=None,
                 **kwargs):

        super(ExperimentLogger, self).__init__(**kwargs)

        self.experiment_name = experiment_name
        self.base_dir = base_dir
        self.folder_format = folder_format

        self.init_time = datetime.datetime.today()

        # try to make folder until successful or until counter runs out
        # this is necessary when multiple ExperimentLoggers start at the same time
        makedir_success = False
        makedir_counter = 100
        makedir_exception = None
        while (not makedir_success and makedir_counter > 0):
            try:
                self.folder_name = self.resolve_format(folder_format, resume)
                self.work_dir = os.path.join(base_dir, self.folder_name)
                if not resume:
                    create_folder(self.work_dir)
                makedir_success = True
            except FileExistsError as file_error:
                makedir_counter -= 1
            except Exception as e:
                makedir_exception = e
                makedir_counter -= 1
        if makedir_exception is not None:
            warnings.warn("Last exception encountered in makedir process:\n" +
                                 "{}\n".format(makedir_exception) +
                                 "There may or may not be a folder for the experiment to run in.", RuntimeWarning)

        self.config_dir = os.path.join(self.work_dir, "config")
        self.log_dir = os.path.join(self.work_dir, "log")
        self.checkpoint_dir = os.path.join(self.work_dir, "checkpoint")
        self.img_dir = os.path.join(self.work_dir, "img")
        self.plot_dir = os.path.join(self.work_dir, "plot")
        self.save_dir = os.path.join(self.work_dir, "save")
        self.result_dir = os.path.join(self.work_dir, "result")

        if not resume:
            create_folder(self.config_dir)
            create_folder(self.log_dir)
            create_folder(self.checkpoint_dir)
            create_folder(self.img_dir)
            create_folder(self.plot_dir)
            create_folder(self.save_dir)
            create_folder(self.result_dir)

        if text_logger_args is None:
            text_logger_args = {}
        if plot_logger_args is None:
            plot_logger_args = {}

        self.text_logger = TextLogger(self.log_dir, **text_logger_args)
        self.plot_logger = NumpyPlotFileLogger(
            self.img_dir, self.plot_dir, **plot_logger_args)

    def show_image(self, image, name, file_format=".png", **kwargs):
        """
        This function shows an image.

        Args:
            image(np.ndarray): image to be shown
            name(str): image title
        """
        self.plot_logger.show_image(image, name, file_format=".png", **kwargs)

    def show_barplot(self, array, name, file_format=".png", **kwargs):
        self.plot_logger.show_barplot(
            array, name, file_format=".png", **kwargs)

    def show_lineplot(self, y_vals, x_vals, name, file_format=".png", **kwargs):
        self.plot_logger.show_lineplot(
            y_vals, x_vals, name, file_format=".png", **kwargs)

    def show_piechart(self, array, name, file_format=".png", **kwargs):
        self.plot_logger.show_piechart(
            array, name, file_format=".png", **kwargs)

    def show_scatterplot(self, array, name, file_format=".png", **kwargs):
        self.plot_logger.show_scatterplot(
            array, name, file_format=".png", **kwargs)

    def show_value(self, value, name=None, counter=None, tag=None, file_format=".png", **kwargs):
        self.plot_logger.show_value(value, name, counter, tag, file_format, **kwargs)

    def show_text(self, text, name=None, logger="default", **kwargs):
        self.text_logger.show_text(text, name, logger, **kwargs)

    def save_model(self):
        raise NotImplementedError

    def load_model(self):
        raise NotImplementedError

    def save_config(self, data, name, **kwargs):

        if not name.endswith(".json"):
            name += ".json"
        data.dump(os.path.join(self.config_dir, name), **kwargs)

    def load_config(self, name, **kwargs):

        if not name.endswith(".json"):
            name += ".json"
        c = Config()
        c.load(os.path.join(self.config_dir, name), **kwargs)
        return c

    def save_checkpoint(self):
        raise NotImplementedError

    def load_checkpoint(self):
        raise NotImplementedError

    def save_result(self, data, name, indent=4, separators=(",", ": "), **kwargs):

        if not name.endswith(".json"):
            name += ".json"
        name = os.path.join(self.result_dir, name)
        create_folder(os.path.dirname(name))
        json.dump(data, open(name, "w"),
                  cls=MultiTypeEncoder,
                  indent=indent,
                  separators=separators,
                  **kwargs)

    def save_dict(self, data, path, indent=4, separators=(",", ": "), **kwargs):

        if not path.endswith(".json"):
            path += ".json"
        path = os.path.join(self.save_dir, path)
        create_folder(os.path.dirname(path))
        json.dump(data, open(path, "w"),
                  cls=MultiTypeEncoder,
                  indent=indent,
                  separators=separators,
                  **kwargs)

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

    def save_file(self, filepath, path=None):

        if path is None:
            target_dir = self.save_dir
        else:
            target_dir = os.path.join(self.save_dir, path)
        filename = os.path.basename(filepath)
        shutil.copy(filepath, os.path.join(target_dir, filename))

    def resolve_format(self, input_, resume):

        if resume:

            pattern = input_[:]

            for find in re.findall("{[\w\:]+}", pattern):
                run_match = re.search("(?<=\{run_number\:0)\d+(?=d\})", find)
                if find == "{run_number}":
                    pattern = pattern.replace("{run_number}", "\d+")
                elif run_match:
                    length = int(run_match.group(0))
                    pattern = re.sub("\{run_number\:\d+d\}",
                                     "\\d{" + str(length) + "}", pattern)
                else:
                    if find[1:-1] in self.__dict__:
                        pattern = pattern.replace(
                            find, self.__dict__[find[1:-1]])

            for r in REPLACEMENTS:
                pattern = pattern.replace(r[0], "\\d{" + str(r[1]) + "}")

            return list(filter(lambda x: re.match(pattern, x),
                               sorted(os.listdir(self.base_dir))))[-1]

        if "%" in input_:
            input_ = self.init_time.strftime(input_)

        if "{" not in input_:
            return input_

        run_number = 1
        while os.path.exists(os.path.join(self.base_dir,
                                          input_.format(run_number=run_number,
                                                        **self.__dict__))):
            # if the folder already exists, for example if two jobs are launched
            # at the same time, append run_number
            if "{run_number}" not in input_:
                input_ = input_ + "_{run_number}"

            run_number += 1

        return input_.format(run_number=run_number, **self.__dict__)
