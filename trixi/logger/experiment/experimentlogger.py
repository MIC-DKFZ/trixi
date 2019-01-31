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

from trixi.logger.abstractlogger import AbstractLogger
from trixi.logger.file.textfilelogger import TextFileLogger
from trixi.logger.file.numpyplotfilelogger import NumpyPlotFileLogger
from trixi.util import create_folder, MultiTypeEncoder, MultiTypeDecoder, Config


REPLACEMENTS = [("%Y", 4), ("%m", 2), ("%d", 2), ("%H", 2), ("%M", 2), ("%S", 2),
                ("%w", 1), ("%y", 2), ("%I", 2), ("%f", 6), ("%j", 3), ("%U", 2),
                ("%W", 2)]


class ExperimentLogger(AbstractLogger):
    """A single class for logging your experiments to file.

    It creates a experiment folder in your base folder and a folder structure to store your experiment files.
    The folder structure is::

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

    def __init__(self,
                 exp_name,
                 base_dir,
                 folder_format="%Y%m%d-%H%M%S_{experiment_name}",
                 resume=False,
                 text_logger_args=None,
                 plot_logger_args=None,
                 **kwargs):
        """
        Initializes the Experiment logger and creates the experiment folder structure

        Args:
            exp_name (str): The name of the experiment
            base_dir (str): The base directory in which the experiment folder will be created
            folder_format (str): The format for the naming of the experiment folder
            resume (bool): if True use the given folder and do not create new ones
            text_logger_args: Parameters for the TextFileLogger initialization
            plot_logger_args: Parameters for the NumpyPlotFileLogger initialization
        """

        super(ExperimentLogger, self).__init__(**kwargs)

        self.experiment_name = exp_name
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

        self.text_logger = TextFileLogger(self.log_dir, **text_logger_args)
        self.plot_logger = NumpyPlotFileLogger(
            self.img_dir, self.plot_dir, **plot_logger_args)

    def show_image(self, image, name, file_format=".png", **kwargs):
        """
        This function saves an image in the experiment img folder.

        Args:
            image(np.ndarray): image to be shown
            name(str): image title
            file_format (str): file format of the image
        """
        self.plot_logger.show_image(image, name, file_format=".png", **kwargs)

    def show_barplot(self, array, name, file_format=".png", **kwargs):
        """
        This function saves a barplot in the experiment plot folder.

        Args:
            array(np.ndarray): array to be plotted
            name(str): image title
            file_format (str): file format of the image
        """
        self.plot_logger.show_barplot(
            array, name, file_format=".png", **kwargs)

    def show_lineplot(self, y_vals, x_vals=None, name="lineplot", file_format=".png", **kwargs):
        """
        This function saves a line plot in the experiment plot folder.

        Args:
            x_vals: x values of the line
            y_vals: y values of the line
            name(str): image title
            file_format (str): file format of the image
        """
        self.plot_logger.show_lineplot(
            y_vals, x_vals, name, file_format=".png", **kwargs)

    def show_piechart(self, array, name, file_format=".png", **kwargs):
        """
        This function saves a piechart in the experiment plot folder.

        Args:
            array(np.ndarray): array to be plotted
            name(str): image title
            file_format (str): file format of the image
        """
        self.plot_logger.show_piechart(
            array, name, file_format=".png", **kwargs)

    def show_scatterplot(self, array, name, file_format=".png", **kwargs):
        """
        This function saves a scatterplot in the experiment plot folder.

        Args:
            array(np.ndarray): array to be plotted
            name(str): image title
            file_format (str): file format of the image
        """
        self.plot_logger.show_scatterplot(
            array, name, file_format=".png", **kwargs)

    def show_value(self, value, name=None, counter=None, tag=None, file_format=".png", **kwargs):
        """
        This function saves a value as a consequtive line plot.

        Args:
            value(np.ndarray): value to be plotted
            name(str): image title
            counter: y-value of the image (if not suppled simply increases for each call)
            tag: group/label for the value. Values with the same tag will be plotted in the same plot
            file_format (str): file format of the image
        """
        self.plot_logger.show_value(value, name, counter, tag, file_format, **kwargs)

    def show_text(self, text, name=None, logger="default", **kwargs):
        """
        Logs a text to a log file.

        Args:
            text: The text to be logged
            name: Name of the text
            logger: log file (in the experiment log folder) in which the text will be logged.
            **kwargs:

        """
        self.text_logger.show_text(text, name, logger, **kwargs)

    def show_boxplot(self, array, name, file_format=".png", **kwargs):
        """
        This function saves a boxplot in the experiment plot folder.

        Args:
            array(np.ndarray): array to be plotted
            name(str): image title
            file_format (str): file format of the image
        """
        self.plot_logger.show_boxplot(
            array, name, file_format=".png", **kwargs)

    def save_model(self):
        raise NotImplementedError

    def load_model(self):
        raise NotImplementedError

    def save_config(self, data, name, **kwargs):
        """
        Saves a config as a json file in the experiment config dir

        Args:
            data: The data to be stored as config json
            name: The name of the json file in which the data will be stored

        """

        if not name.endswith(".json"):
            name += ".json"
        data.dump(os.path.join(self.config_dir, name), **kwargs)

    def load_config(self, name, **kwargs):
        """
        Loads a config from a json file from the experiment config dir

        Args:
            name: the name of the config file

        Returns: A Config/ dict filled with the json file content

        """

        if not name.endswith(".json"):
            name += ".json"
        c = Config()
        c.load(os.path.join(self.config_dir, name), **kwargs)
        return c

    def save_checkpoint(self):
        raise NotImplementedError

    def load_checkpoint(self):
        raise NotImplementedError

    def save_result(self, data, name, indent=4, separators=(",", ": "), encoder_cls=MultiTypeEncoder, **kwargs):
        """
        Saves data as a json file in the experiment result dir

        Args:
            data: The data to be stored as result json
            name: name of the result json file
            indent: Indent for the json file
            separators: Separators for the json file
            encoder_cls: Encoder Class for the encoding to json

        """

        if not name.endswith(".json"):
            name += ".json"
        name = os.path.join(self.result_dir, name)
        create_folder(os.path.dirname(name))
        with open(name, "w") as jf:
            json.dump(data, jf,
                      cls=encoder_cls,
                      indent=indent,
                      separators=separators,
                      **kwargs)

    def save_dict(self, data, path, indent=4, separators=(",", ": "), encoder_cls=MultiTypeEncoder, **kwargs):
        """
        Saves a dict as a json file in the experiment save dir

        Args:
            data: The data to be stored as save file
            path: sub path in the save folder (or simply filename)
            indent: Indent for the json file
            separators: Separators for the json file
            encoder_cls: Encoder Class for the encoding to json
        """

        if not path.endswith(".json"):
            path += ".json"
        path = os.path.join(self.save_dir, path)
        create_folder(os.path.dirname(path))
        with open(path, "w") as jf:
            json.dump(data, jf,
                      cls=encoder_cls,
                      indent=indent,
                      separators=separators,
                      **kwargs)

    def load_dict(self, path):
        """
        Loads a json file as dict from a sub path in the experiment save dir

        Args:
            path: sub path to the file (starting from the experiment save dir)

        Returns: The restored data as a dict
        """

        if not path.endswith(".json"):
            path += ".json"
        path = os.path.join(self.save_dir, path)
        ret_val = dict()
        with open(path, "r") as df:
            ret_val = json.load(df, cls=MultiTypeDecoder)
        return ret_val

    def save_numpy_data(self, data, path):
        """
            Saves a numpy array in the experiment save dir

            Args:
                data: The array to be stored as a save file
                path: sub path in the save folder (or simply filename)
        """

        if not path.endswith(".npy"):
            path += ".npy"
        path = os.path.join(self.save_dir, path)
        create_folder(os.path.dirname(path))
        np.save(path, data)

    def load_numpy_data(self, path):
        """
        Loads a numpy file from a sub path in the experiment save dir

        Args:
            path: sub path to the file (starting from the experiment save dir)

        Returns: The restored numpy array
        """

        if not path.endswith(".npy"):
            path += ".npy"
        path = os.path.join(self.save_dir, path)
        return np.load(path)

    def save_pickle(self, data, path):
        """
            Saves a object data in the experiment save dir via pickle

            Args:
                data: The data to be stored as a save file
                path: sub path in the save folder (or simply filename)
        """

        path = os.path.join(self.save_dir, path)
        create_folder(os.path.dirname(path))
        with open(path, "wb") as out:
            pickle.dump(data, out)

    def load_pickle(self, path):
        """
        Loads a object via pickle from a sub path in the experiment save dir

        Args:
            path: sub path to the file (starting from the experiment save dir)

        Returns: The restored object
        """

        path = os.path.join(self.save_dir, path)
        with open(path, "rb") as in_:
            return pickle.load(in_)

    def save_file(self, filepath, path=None):
        """
        Copies a file to the experiment save dir

        Args:
            filepath: Path to the file to be copied to the experiment save dir
            path: sub path to the target file (starting from the experiment save dir, does not have to exist yet)

        """

        if path is None:
            target_dir = self.save_dir
        else:
            target_dir = os.path.join(self.save_dir, path)
        filename = os.path.basename(filepath)
        shutil.copy(filepath, os.path.join(target_dir, filename))

    def resolve_format(self, input_, resume):
        """
        Given some input pattern, tries to find the best matching folder name by resolving the format. Options are:
         - Run-number: {run_number}
         - Time: "%Y%m%d-%H%M%S
         - Member variables (e.g experiment_name) : {variable_name} (e.g. {experiment_name})


        Args:
            input_: The format to be resolved
            resume: Flag if folder should be resumed

        Returns: The resolved folder name

        """

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
