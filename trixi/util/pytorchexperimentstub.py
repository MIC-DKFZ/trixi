import json
import os
import time
import warnings
from unittest.mock import Mock

import numpy as np

from trixi.logger import PytorchExperimentLogger, PytorchVisdomLogger, TelegramMessageLogger
from trixi.logger.message.slackmessagelogger import SlackMessageLogger
from trixi.logger.tensorboard import PytorchTensorboardXLogger
from trixi.util import ResultElement, ResultLogDict, Config

logger_lookup_dict = dict(
    visdom=PytorchVisdomLogger,
    tensorboard=PytorchTensorboardXLogger,
    telegram=TelegramMessageLogger,
    slack=SlackMessageLogger,
)


class PytorchExperimentStub:

    def __init__(self, base_dir=None, name=None, config=None, loggers=None):
        super(PytorchExperimentStub, self).__init__()

        if config is None:
            config = {}
        if loggers is None:
            loggers = {}

        # assert base_dir is not None or "base_dir" in config, "A base dir has to be given, either directly or via config"

        if name is None and 'name' in config:
            self.name = config['name']
        elif name is None:
            self.name = "experiment"
        else:
            self.name = name

        if base_dir is not None:
            self.base_dir = base_dir
        else:
            self.base_dir = config.get('base_dir')

        self.config = config

        if base_dir is not None:
            self.elog = PytorchExperimentLogger(base_dir=self.base_dir,
                                                exp_name=self.name)

            self.results = ResultLogDict("results-log.json", base_dir=self.elog.result_dir)
        else:
            warnings.warn("PytorchExperimentStub will not save to drive")
            self.elog = Mock()
            self.results = dict()

        self.loggers = {}
        for logger_name, logger_cfg in loggers.items():
            _logger = self._make_logger(logger_name, logger_cfg)
            self.loggers[logger_name] = _logger

        self._save_exp_config()
        self.elog.save_config(self.config, "config")

    def _make_logger(self, logger_name, logger_cfg):

        if isinstance(logger_cfg, (list, tuple)):
            log_type = logger_cfg[0]
            log_params = logger_cfg[1] if len(logger_cfg) > 1 else {}
            log_freq = logger_cfg[2] if len(logger_cfg) > 2 else 10
        else:
            assert isinstance(logger_cfg, str), "The specified logger has to either be a string or a list with " \
                                                "name, parameters, clog_frequency"
            log_type = logger_cfg
            log_params = {}
            log_freq = 10

        if "exp_name" not in log_params:
            log_params["exp_name"] = self.name

        if log_type == "tensorboard":
            if "target_dir" not in log_params or log_params["target_dir"] is None:
                if self.elog is not None and not isinstance(self.elog, Mock):
                    log_params["target_dir"] = os.path.join(self.elog.save_dir, "tensorboard")
                else:
                    raise AttributeError("TensorboardLogger requires a target_dir or an ExperimentLogger instance.")
            elif self.elog is not None and not isinstance(self.elog, Mock):
                log_params["target_dir"] = os.path.join(log_params["target_dir"], self.elog.folder_name)

        log_type = logger_lookup_dict[log_type]
        _logger = log_type(**log_params)

        return _logger

    @property
    def vlog(self):
        if "visdom" in self.loggers:
            return self.loggers["visdom"]
        elif "v" in self.loggers:
            return self.loggers["v"]
        else:
            return None

    @property
    def tlog(self):
        if "telegram" in self.loggers:
            return self.loggers["telegram"]
        elif "t" in self.loggers:
            return self.loggers["t"]
        else:
            return None

    @property
    def txlog(self):
        if "tensorboard" in self.loggers:
            return self.loggers["tensorboard"]
        if "tensorboardx" in self.loggers:
            return self.loggers["tensorboardx"]
        elif "tx" in self.loggers:
            return self.loggers["tx"]
        else:
            return None

    @property
    def slog(self):
        if "slack" in self.loggers:
            return self.loggers["slack"]
        elif "s" in self.loggers:
            return self.loggers["s"]
        else:
            return None

    def _save_exp_config(self):

        if self.elog is not None and not isinstance(self.elog, Mock):
            cur_time = time.strftime("%y-%m-%d_%H:%M:%S", time.localtime(time.time()))
            self.elog.save_config(Config(**{'name': self.name,
                                            'time': cur_time,
                                            'state': "Stub",
                                            'current_time': cur_time,
                                            'epoch': 0
                                            }),
                                  "exp")

    def add_result(self, value, name, counter=None, tag=None, label=None, plot_result=True, plot_running_mean=False):
        """
        Saves a results and add it to the result dict, this is similar to results[key] = val,
        but in addition also logs the value to the combined logger
        (it also stores in the results-logs file).

        **This should be your preferred method to log your numeric values**

        Args:
            value: The value of your variable
            name (str): The name/key of your variable
            counter (int or float): A counter which can be seen as the x-axis of your value.
                Normally you would just use the current epoch for this.
            tag (str): A label/tag which can group similar values and will plot values with the same
                label in the same plot
            label: deprecated label
            plot_result (bool): By default True, will also log all your values to the combined
                logger (with show_value).

        """

        if label is not None:
            warnings.warn("label in add_result is deprecated, please use tag instead")

            if tag is None:
                tag = label

        tag_name = tag
        if tag_name is None:
            tag_name = name

        r_elem = ResultElement(data=value, label=tag_name, epoch=0, counter=counter)

        self.results[name] = r_elem

        if plot_result:
            if tag is None:
                legend = False
            else:
                legend = True
            if plot_running_mean:
                value = np.mean(self.results.running_mean_dict[name])
            self.elog.show_value(value=value, name=name, tag=tag_name, counter=counter, show_legend=legend)
            if "visdom" in self.loggers:
                self.vlog.show_value(value=value, name=name, tag=tag_name, counter=counter, show_legend=legend)
            if "tensorboard" in self.loggers:
                self.txlog.show_value(value=value, name=name, tag=tag_name, counter=counter, show_legend=legend)

    def get_result(self, name):
        """
        Similar to result[key] this will return the values in the results dictionary with the given
        name/key.

        Args:
            name (str): the name/key for which a value is stored.

        Returns:
            The value with the key 'name' in the results dict.

        """
        return self.results.get(name)

    def add_result_without_epoch(self, val, name):
        """
        A faster method to store your results, has less overhead and does not call the combined
        logger. Will only store to the results dictionary.

        Args:
            val: the value you want to add.
            name (str): the name/key of your value.

        """
        self.results[name] = val

    def add_res(self, **kwargs):
        """
                A faster method to store your results, has less overhead and does not call the combined
                logger. Will only store to the results dictionary.

                Args:
                    kwargs: dict with name/keys of your values the values you want to add.
                """
        for key, val in kwargs.items():
            self.results[key] = val

    def get_result_without_epoch(self, name):
        """
        Similar to result[key] this will return the values in result with the given name/key.

        Args:
            name (str): the name/ key for which a value is stores.

        Returns:
            The value with the key 'name' in the results dict.

        """
        return self.results.get(name)

    def print(self, *args):
        """
        Calls 'print' on the experiment logger or uses builtin 'print' if former is not
        available.
        """

        if self.elog is None or isinstance(self.elog, Mock):
            print(*args)
        else:
            self.elog.print(*args)

    def save_results(self, name="results.json"):
        """
        Saves the result dict as a json file in the result dir of the experiment logger.

        Args:
            name (str): The name of the json file in which the results are written.

        """
        if self.elog is None or isinstance(self.elog, Mock):
            return
        with open(os.path.join(self.elog.result_dir, name), "w") as file_:
            json.dump(self.results, file_, indent=4)

    def close_tmp_results(self):
        """
        Closes the tmp results file (i.e. add a '}' )

        """
        if isinstance(self.results, ResultLogDict):
            self.results.close()

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
        self.elog.save_model(model, name, n_iter, iter_format, prefix)

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
        self.elog.save_checkpoint(name, n_iter, iter_format, prefix, **kwargs)
