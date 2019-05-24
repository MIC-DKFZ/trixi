import atexit
import fnmatch
import json
import os
import numpy as np
import sys
import re
import random
import shutil
import string
import time
import traceback
import warnings

import torch

from trixi.experiment.experiment import Experiment
from trixi.logger import CombinedLogger, PytorchExperimentLogger, PytorchVisdomLogger


from trixi.logger.tensorboard.pytorchtensorboardxlogger import PytorchTensorboardXLogger
from trixi.util import Config, ResultElement, ResultLogDict, SourcePacker, name_and_iter_to_filename
from trixi.util.config import update_from_sys_argv
from trixi.util.pytorchutils import set_seed

logger_lookup_dict = dict(
    visdom=PytorchVisdomLogger,
    tensorboard=PytorchTensorboardXLogger,
)

try:
    from trixi.logger import TelegramMessageLogger
    from trixi.logger.message.slackmessagelogger import SlackMessageLogger
    logger_lookup_dict["slack"] = SlackMessageLogger
    logger_lookup_dict["telegram"] = TelegramMessageLogger
except:
    pass


class PytorchExperiment(Experiment):
    """
    A PytorchExperiment extends the basic
    functionality of the :class:`.Experiment` class with
    convenience features for PyTorch (and general logging) such as creating a folder structure,
    saving, plotting results and checkpointing your experiment.

    The basic life cycle of a PytorchExperiment is the same as
    :class:`.Experiment`::

        setup()
        prepare()

        for epoch in n_epochs:
            train()
            validate()

        end()

    where the distinction between the first two is that between them
    PytorchExperiment will automatically restore checkpoints and save the
    :attr:`_config_raw` in :meth:`._setup_internal`. Please see below for more
    information on this.
    To get your own experiment simply inherit from the PytorchExperiment and
    overwrite the :meth:`.setup`, :meth:`.prepare`, :meth:`.train`,
    :meth:`.validate` method (or you can use the `very` experimental decorator
    :func:`.experimentify` to convert your class into a experiment).
    Then you can run your own experiment by calling the :meth:`.run` method.

    Internally PytorchExperiment will provide a number of member variables which
    you can access.

        - n_epochs
            Number of epochs.
        - exp_name
            Name of your experiment.
        - config
            The (initialized) :class:`.Config` of your experiment. You can
            access the uninitialized one via :attr:`_config_raw`.
        - result
            A dict in which you can store your result values. If a
            :class:`.PytorchExperimentLogger` is used, results will be a
            :class:`.ResultLogDict` that directly automatically writes to a file
            and also stores the N last entries for each key for quick access
            (e.g. to quickly get the running mean).
        - elog (if base_dir is given)
            A :class:`.PytorchExperimentLogger` instance which can log your
            results to a given folder. Will automatically be created if a base_dir is available.
        - loggers
            Contains all loggers you provide, including the experiment logger, accessible by the
            names you provide.
        - clog
            A :class:`.CombinedLogger` instance which logs to all loggers with
            different frequencies (specified with the last entry in the tuple you provide for each
            logger where 1 means every time and N means every Nth time,
            e.g. if you only want to send stuff to Visdom every 10th time).

    The most important attribute is certainly :attr:`.config`, which is the
    initialized :class:`.Config` for the experiment. To understand how it needs
    to be structured to allow for automatic instantiation of types, please refer
    to its documentation. If you decide not to use this functionality,
    :attr:`config` and :attr:`_config_raw` are identical. **Beware however that
    by default the Pytorchexperiment only saves the raw config** after
    :meth:`.setup`. If you modify :attr:`config` during setup, make sure
    to implement :meth:`._setup_internal` yourself should you want the modified
    config to be saved::

        def _setup_internal(self):

            super(YourExperiment, self)._setup_internal() # calls .prepare_resume()
            self.elog.save_config(self.config, "config")

    Args:
        config (dict or Config): Configures your experiment. If :attr:`name`,
            :attr:`n_epochs`, :attr:`seed`, :attr:`base_dir` are given in the
            config, it will automatically
            overwrite the other args/kwargs with the values from the config.
            In addition (defined by :attr:`parse_config_sys_argv`) the config
            automatically parses the argv arguments and updates its values if a
            key matches a console argument.
        name (str):
            The name of the PytorchExperiment.
        n_epochs (int): The number of epochs (number of times the training
            cycle will be executed).
        seed (int): A random seed (which will set the random, numpy and
            torch seed).
        base_dir (str): A base directory in which the experiment result folder
            will be created. A :class:`.PytorchExperimentLogger` instance will be created if
            this is given.
        globs: The :func:`globals` of the script which is run. This is necessary
            to get and save the executed files in the experiment folder.
        resume (str or PytorchExperiment): Another PytorchExperiment or path to
            the result dir from another PytorchExperiment from which it will
            load the PyTorch modules and other member variables and resume
            the experiment.
        ignore_resume_config (bool): If :obj:`True` it will not resume with the
            config from the resume experiment but take the current/own config.
        resume_save_types (list or tuple): A list which can define which values
            to restore when resuming. Choices are:

                - "model" <-- Pytorch models
                - "optimizer" <-- Optimizers
                - "simple" <-- Simple python variables (basic types and lists/tuples
                - "th_vars" <-- torch tensors/variables
                - "results" <-- The result dict

        resume_reset_epochs (bool): Set epoch to zero if you resume an existing experiment.
        parse_sys_argv (bool): Parsing the console arguments (argv) to get a
            :attr:`config path` and/or :attr:`resume_path`.
        parse_config_sys_argv (bool): Parse argv to update the config
            (if the keys match).
        checkpoint_to_cpu (bool): When checkpointing, transfer all tensors to
            the CPU beforehand.
        save_checkpoint_every_epoch (int): Determines after how many epochs a
            checkpoint is stored.
        explogger_kwargs (dict): Keyword arguments for :attr:`elog`
            instantiation.
        explogger_freq (int): The frequency x (meaning one in x) with which
            the :attr:`clog` will call the :attr:`elog`.
        loggers (dict): Specify additional loggers.
            Entries should have one of these formats::

                "name": "identifier" (will default to a frequency of 10)
                "name": ("identifier"(, kwargs, frequency)) (last two are optional)

            "identifier" is one of "telegram", "tensorboard", "visdom", "slack".
        append_rnd_to_name (bool): If :obj:`True`, will append a random six
            digit string to the experiment name.

     """

    def __init__(self,
                 config=None,
                 name=None,
                 n_epochs=None,
                 seed=None,
                 base_dir=None,
                 globs=None,
                 resume=None,
                 ignore_resume_config=False,
                 resume_save_types=("model", "optimizer", "simple", "th_vars", "results"),
                 resume_reset_epochs=True,
                 parse_sys_argv=False,
                 checkpoint_to_cpu=True,
                 save_checkpoint_every_epoch=1,
                 explogger_kwargs=None,
                 explogger_freq=1,
                 loggers=None,
                 append_rnd_to_name=False,
                 default_save_types=("model", "optimizer", "simple", "th_vars", "results")):

        # super(PytorchExperiment, self).__init__()
        Experiment.__init__(self)

        # check for command line inputs for config_path and resume_path,
        # will be prioritized over config and resume!
        config_path_from_argv = None
        if parse_sys_argv:
            config_path_from_argv, resume_path_from_argv = get_vars_from_sys_argv()
            if resume_path_from_argv:
                resume = resume_path_from_argv

        # construct _config_raw
        if config_path_from_argv is None:
            self._config_raw = self._config_raw_from_input(config, name, n_epochs, seed, append_rnd_to_name)
        else:
            self._config_raw = Config(file_=config_path_from_argv)
        update_from_sys_argv(self._config_raw)

        # set a few experiment attributes
        self.n_epochs = self._config_raw["n_epochs"]
        self._seed = self._config_raw['seed']
        set_seed(self._seed)
        self.exp_name = self._config_raw["name"]
        self._checkpoint_to_cpu = checkpoint_to_cpu
        self._save_checkpoint_every_epoch = save_checkpoint_every_epoch
        self._default_save_types = ("model", "optimizer", "simple", "th_vars", "results")
        self.results = dict()

        # get base_dir from _config_raw or store there
        if base_dir is not None:
            self._config_raw["base_dir"] = base_dir
        base_dir = self._config_raw["base_dir"]

        # Construct experiment logger (automatically activated if base_dir is there)
        self.loggers = {}
        logger_list = []
        if base_dir is not None:
            if explogger_kwargs is None:
                explogger_kwargs = {}
            self.elog = PytorchExperimentLogger(base_dir=base_dir,
                                                exp_name=self.exp_name,
                                                **explogger_kwargs)
            if explogger_freq is not None and explogger_freq > 0:
                logger_list.append((self.elog, explogger_freq))
            self.results = ResultLogDict("results-log.json", base_dir=self.elog.result_dir)
        else:
            self.elog = None

        # Construct other loggers
        if loggers is not None:
            for logger_name, logger_cfg in loggers.items():
                _logger, log_freq = self._make_logger(logger_name, logger_cfg)
                self.loggers[logger_name] = _logger
                if log_freq is not None and log_freq > 0:
                    logger_list.append((_logger, log_freq))

        self.clog = CombinedLogger(*logger_list)

        # Set resume attributes and update _config_raw,
        # actual resuming is done automatically after setup in _setup_internal
        self._resume_path = None
        self._resume_save_types = resume_save_types
        self._ignore_resume_config = ignore_resume_config
        self._resume_reset_epochs = resume_reset_epochs
        if resume is not None:
            if isinstance(resume, str):
                if resume == "last":
                    if base_dir is None:
                        raise ValueError("resume='last' requires base_dir.")
                    self._resume_path = os.path.join(base_dir, sorted(os.listdir(base_dir))[-1])
                else:
                    self._resume_path = resume
            elif isinstance(resume, PytorchExperiment):
                self._resume_path = resume.elog.base_dir
        if self._resume_path is not None and not self._ignore_resume_config:
            self._config_raw.update(Config(file_=os.path.join(self._resume_path, "config", "config.json")),
                                    ignore=list(map(lambda x: re.sub("^-+", "", x), sys.argv)))

        # Save everything we need to reproduce experiment
        if globs is not None and self.elog is not None:
            zip_name = os.path.join(self.elog.save_dir, "sources.zip")
            SourcePacker.zip_sources(globs, zip_name)

        # Init objects in config
        self.config = Config.init_objects(self._config_raw)

        atexit.register(self.at_exit_func)

    def _config_raw_from_input(self,
                               config=None,
                               name=None,
                               n_epochs=None,
                               seed=None,
                               append_rnd_to_name=False):
        """Construct _config_raw from input."""

        _config_raw = None
        if isinstance(config, str):
            _config_raw = Config(file_=config)
        elif isinstance(config, (Config, dict)):
            _config_raw = Config(config=config)
        else:
            _config_raw = Config()

        if n_epochs is None and _config_raw.get("n_epochs") is not None:
            n_epochs = _config_raw["n_epochs"]
        elif n_epochs is None and _config_raw.get("n_epochs") is None:
            n_epochs = 0
        _config_raw["n_epochs"] = n_epochs

        if seed is None and _config_raw.get('seed') is not None:
            seed = _config_raw['seed']
        elif seed is None and _config_raw.get('seed') is None:
            random_data = os.urandom(4)
            seed = int.from_bytes(random_data, byteorder="big")
        _config_raw['seed'] = seed

        if name is None and _config_raw.get("name") is not None:
            name = _config_raw["name"]
        elif name is None and _config_raw.get("name") is None:
            name = "experiment"
        if append_rnd_to_name:
            rnd_str = ''.join(random.choice(string.ascii_letters + string.digits) for _ in range(5))
            name += "_" + rnd_str
        _config_raw["name"] = name

        return _config_raw

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
            log_params["exp_name"] = self.exp_name

        if log_type == "tensorboard":
            if "target_dir" not in log_params or log_params["target_dir"] is None:
                if self.elog is not None:
                    log_params["target_dir"] = os.path.join(self.elog.save_dir, "tensorboard")
                else:
                    raise AttributeError("TensorboardLogger requires a target_dir or an ExperimentLogger instance.")
            elif self.elog is not None:
                log_params["target_dir"] = os.path.join(log_params["target_dir"], self.elog.folder_name)

        log_type = logger_lookup_dict[log_type]
        _logger = log_type(**log_params)

        return _logger, log_freq

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

    def process_err(self, e):
        if self.elog is not None:
            self.elog.text_logger.log_to("\n".join(traceback.format_tb(e.__traceback__)), "err")
            self.elog.text_logger.log_to(repr(e), "err")
        raise e

    def update_attributes(self, var_dict, ignore=()):
        """
        Updates the member attributes with the attributes given in the var_dict

        Args:
            var_dict (dict): dict in which the update values stored. If a key matches a member attribute name
                the member attribute will be updated
            ignore (list or tuple): iterable of keys to ignore

        """
        for key, val in var_dict.items():
            if key == "results":
                self.results.load(val)
                continue
            if key in ignore:
                continue
            if hasattr(self, key):
                setattr(self, key, val)

    def get_pytorch_modules(self, from_config=True):
        """
        Returns all torch.nn.Modules stored in the experiment in a dict (even child dicts are stored).

        Args:
            from_config (bool): Also get modules that are stored in the :attr:`.config` attribute.

        Returns:
            dict: Dictionary of PyTorch modules

        """

        def parse_torchmodules_recursive(input, output):
            if isinstance(input, dict):
                for key, value in input.items():
                    if isinstance(value, dict):
                        parse_torchmodules_recursive(value, output)
                    elif isinstance(value, torch.nn.Module):
                        output[key] = value

        pyth_modules = dict()
        parse_torchmodules_recursive(self.__dict__, pyth_modules)
        # for key, val in self.__dict__.items():
        #     if isinstance(val, torch.nn.Module):
        #         pyth_modules[key] = val
        if from_config:
            for key, val in self.config.items():
                if isinstance(val, torch.nn.Module):
                    if type(key) == str:
                        key = "config." + key
                    pyth_modules[key] = val
        return pyth_modules

    def get_pytorch_optimizers(self, from_config=True):
        """
        Returns all torch.optim.Optimizers stored in the experiment in a dict.

        Args:
            from_config (bool): Also get optimizers that are stored in the :attr:`.config`
                attribute.

        Returns:
            dict: Dictionary of PyTorch optimizers

        """

        pyth_optimizers = dict()
        for key, val in self.__dict__.items():
            if isinstance(val, torch.optim.Optimizer):
                pyth_optimizers[key] = val
        if from_config:
            for key, val in self.config.items():
                if isinstance(val, torch.optim.Optimizer):
                    if type(key) == str:
                        key = "config." + key
                    pyth_optimizers[key] = val
        return pyth_optimizers

    def get_simple_variables(self, ignore=()):
        """
        Returns all standard variables in the experiment in a dict.
        Specifically, this looks for types :class:`int`, :class:`float`, :class:`bytes`,
        :class:`bool`, :class:`str`, :class:`set`, :class:`list`, :class:`tuple`.

        Args:
            ignore (list or tuple): Iterable of names which will be ignored

        Returns:
            dict: Dictionary of variables

        """

        simple_vars = dict()
        for key, val in self.__dict__.items():
            if key in ignore:
                continue
            if isinstance(val, (int, float, bytes, bool, str, set, list, tuple)):
                simple_vars[key] = val
        return simple_vars

    def get_pytorch_tensors(self, ignore=()):
        """
        Returns all torch.tensors in the experiment in a dict.

        Args:
            ignore (list or tuple): Iterable of names which will be ignored

        Returns:
            dict: Dictionary of PyTorch tensor

        """

        pytorch_vars = dict()
        for key, val in self.__dict__.items():
            if key in ignore:
                continue
            if torch.is_tensor(val):
                pytorch_vars[key] = val
        return pytorch_vars

    def get_pytorch_variables(self, ignore=()):
        """Same as :meth:`.get_pytorch_tensors`."""
        return self.get_pytorch_tensors(ignore)

    def save_results(self, name="results.json"):
        """
        Saves the result dict as a json file in the result dir of the experiment logger.

        Args:
            name (str): The name of the json file in which the results are written.

        """
        if self.elog is None:
            return
        with open(os.path.join(self.elog.result_dir, name), "w") as file_:
            json.dump(self.results, file_, indent=4)

    def save_pytorch_models(self):
        """Saves all torch.nn.Modules as model files in the experiment checkpoint folder."""

        if self.elog is None:
            return

        pyth_modules = self.get_pytorch_modules()
        for key, val in pyth_modules.items():
            self.elog.save_model(val, key)

    def load_pytorch_models(self):
        """Loads all model files from the experiment checkpoint folder."""

        if self.elog is None:
            return
        pyth_modules = self.get_pytorch_modules()
        for key, val in pyth_modules.items():
            self.elog.load_model(val, key)

    def log_simple_vars(self):
        """
        Logs all simple python member variables as a json file in the experiment log folder.
        The file will be names 'simple_vars.json'.
        """

        if self.elog is None:
            return
        simple_vars = self.get_simple_variables()
        with open(os.path.join(self.elog.log_dir, "simple_vars.json"), "w") as file_:
            json.dump(simple_vars, file_)

    def load_simple_vars(self):
        """
        Restores all simple python member variables from the 'simple_vars.json' file in the log
        folder.
        """

        if self.elog is None:
            return
        simple_vars = {}
        with open(os.path.join(self.elog.log_dir, "simple_vars.json"), "r") as file_:
            simple_vars = json.load(file_)
        self.update_attributes(simple_vars)

    def save_checkpoint(self,
                        name="checkpoint",
                        save_types=("model", "optimizer", "simple", "th_vars", "results"),
                        n_iter=None,
                        iter_format="{:05d}",
                        prefix=False):
        """
        Saves a current model checkpoint from the experiment.

        Args:
            name (str): The name of the checkpoint file
            save_types (list or tuple): What kind of member variables should be stored? Choices are:
                "model" <-- Pytorch models,
                "optimizer" <-- Optimizers,
                "simple" <-- Simple python variables (basic types and lists/tuples),
                "th_vars" <-- torch tensors,
                "results" <-- The result dict
            n_iter (int): Number of iterations. Together with the name, defined by the iter_format,
                a file name will be created.
            iter_format (str): Defines how the name and the n_iter will be combined.
            prefix (bool): If True, the formatted n_iter will be prepended, otherwise appended.

        """

        if self.elog is None:
            return

        model_dict = {}
        optimizer_dict = {}
        simple_dict = {}
        th_vars_dict = {}
        results_dict = {}

        if "model" in save_types:
            model_dict = self.get_pytorch_modules()
        if "optimizer" in save_types:
            optimizer_dict = self.get_pytorch_optimizers()
        if "simple" in save_types:
            simple_dict = self.get_simple_variables()
        if "th_vars" in save_types:
            th_vars_dict = self.get_pytorch_variables()
        if "results" in save_types:
            results_dict = {"results": self.results}

        checkpoint_dict = {**model_dict, **optimizer_dict, **simple_dict, **th_vars_dict, **results_dict}

        self.elog.save_checkpoint(name=name, n_iter=n_iter, iter_format=iter_format, prefix=prefix,
                                  move_to_cpu=self._checkpoint_to_cpu, **checkpoint_dict)

    def load_checkpoint(self,
                        name="checkpoint",
                        save_types=("model", "optimizer", "simple", "th_vars", "results"),
                        n_iter=None,
                        iter_format="{:05d}",
                        prefix=False,
                        path=None):
        """
        Loads a checkpoint and restores the experiment.

        Make sure you have your torch stuff already on the right devices beforehand,
        otherwise this could lead to errors e.g. when making a optimizer step
        (and for some reason the Adam states are not already on the GPU:
        https://discuss.pytorch.org/t/loading-a-saved-model-for-continue-training/17244/3 )

        Args:
            name (str): The name of the checkpoint file
            save_types (list or tuple): What kind of member variables should be loaded? Choices are:
                "model" <-- Pytorch models,
                "optimizer" <-- Optimizers,
                "simple" <-- Simple python variables (basic types and lists/tuples),
                "th_vars" <-- torch tensors,
                "results" <-- The result dict
            n_iter (int): Number of iterations. Together with the name, defined by the iter_format,
                a file name will be created and searched for.
            iter_format (str): Defines how the name and the n_iter will be combined.
            prefix (bool): If True, the formatted n_iter will be prepended, otherwise appended.
            path (str): If no path is given then it will take the current experiment dir and formatted
                name, otherwise it will simply use the path and the formatted name to define the
                checkpoint file.

        """
        if self.elog is None:
            return

        model_dict = {}
        optimizer_dict = {}
        simple_dict = {}
        th_vars_dict = {}
        results_dict = {}

        if "model" in save_types:
            model_dict = self.get_pytorch_modules()
        if "optimizer" in save_types:
            optimizer_dict = self.get_pytorch_optimizers()
        if "simple" in save_types:
            simple_dict = self.get_simple_variables()
        if "th_vars" in save_types:
            th_vars_dict = self.get_pytorch_variables()
        if "results" in save_types:
            results_dict = {"results": self.results}

        checkpoint_dict = {**model_dict, **optimizer_dict, **simple_dict, **th_vars_dict, **results_dict}

        if n_iter is not None:
            name = name_and_iter_to_filename(name,
                                             n_iter,
                                             ".pth.tar",
                                             iter_format=iter_format,
                                             prefix=prefix)

        if path is None:
            restore_dict = self.elog.load_checkpoint(name=name, **checkpoint_dict)
        else:
            checkpoint_path = os.path.join(path, name)
            if checkpoint_path.endswith(os.sep):
                checkpoint_path = os.path.dirname(checkpoint_path)
            restore_dict = self.elog.load_checkpoint_static(checkpoint_file=checkpoint_path, **checkpoint_dict)

        self.update_attributes(restore_dict)

    def _end_internal(self):
        """Ends the experiment and stores the final results/checkpoint"""
        if isinstance(self.results, ResultLogDict):
            self.results.close()
        self.save_results()
        self.save_end_checkpoint()
        self._save_exp_config()
        self.print("Experiment ended. Checkpoints stored =)")

    def _end_test_internal(self):
        """Ends the experiment after test and stores the final results and config"""
        self.save_results()
        self._save_exp_config()
        self.print("Testing ended. Results stored =)")

    def at_exit_func(self):
        """
        Stores the results and checkpoint at the end (if not already stored).
        This method is also called if an error occurs.
        """

        if self._exp_state not in ("Ended", "Tested"):
            if isinstance(self.results, ResultLogDict):
                self.results.print_to_file("]")
            self.save_checkpoint(name="checkpoint_exit-" + self._exp_state, save_types=self._default_save_types)
            self.save_results()
            self._save_exp_config()
            self.print("Experiment exited. Checkpoints stored =)")
        time.sleep(2)  # allow checkpoint saving to finish. We need a better solution for this :D

    def _setup_internal(self):
        self.prepare_resume()

        if self.elog is not None:
            self.elog.save_config(self._config_raw, "config")
        self._save_exp_config()

    def _start_internal(self):
        self._save_exp_config()

    def prepare_resume(self):
        """Tries to resume the experiment by using the defined resume path or PytorchExperiment."""

        checkpoint_file = ""
        base_dir = ""

        reset_epochs = self._resume_reset_epochs

        if self._resume_path is not None:
            if isinstance(self._resume_path, str):
                if self._resume_path.endswith(".pth.tar"):
                    checkpoint_file = self._resume_path
                    base_dir = os.path.dirname(os.path.dirname(checkpoint_file))
                elif self._resume_path.endswith("checkpoint") or self._resume_path.endswith("checkpoint/"):
                    checkpoint_file = get_last_file(self._resume_path)
                    base_dir = os.path.dirname(os.path.dirname(checkpoint_file))
                elif "checkpoint" in os.listdir(self._resume_path) and "config" in os.listdir(self._resume_path):
                    checkpoint_file = get_last_file(self._resume_path)
                    base_dir = self._resume_path
                else:
                    warnings.warn("You have not selected a valid experiment folder, will search all sub folders",
                                  UserWarning)
                    if self.elog is not None:
                        self.elog.text_logger.log_to("You have not selected a valid experiment folder, will search all "
                                                     "sub folders", "warnings")
                    checkpoint_file = get_last_file(self._resume_path)
                    base_dir = os.path.dirname(os.path.dirname(checkpoint_file))

        if base_dir:
            if not self._ignore_resume_config:
                load_config = Config()
                load_config.load(os.path.join(base_dir, "config/config.json"))
                self._config_raw = load_config
                self.config = Config.init_objects(self._config_raw)
                self.print("Loaded existing config from:", base_dir)
                if self.n_epochs is None:
                    self.n_epochs = self._config_raw.get("n_epochs")

        if checkpoint_file:
            self.load_checkpoint(name="", path=checkpoint_file, save_types=self._resume_save_types)
            self._resume_path = checkpoint_file
            shutil.copyfile(checkpoint_file, os.path.join(self.elog.checkpoint_dir, "0_checkpoint.pth.tar"))
            self.print("Loaded existing checkpoint from:", checkpoint_file)

            self._resume_reset_epochs = reset_epochs
            if self._resume_reset_epochs:
                self._epoch_idx = 0

    def _end_epoch_internal(self, epoch):
        self.save_results()
        if self._save_checkpoint_every_epoch is not None and self._save_checkpoint_every_epoch > 0 and epoch % \
                self._save_checkpoint_every_epoch == 0:
            self.save_temp_checkpoint()
        self._save_exp_config()

    def _save_exp_config(self):

        if self.elog is not None:
            cur_time = time.strftime("%y-%m-%d_%H:%M:%S", time.localtime(time.time()))
            self.elog.save_config(Config(**{'name': self.exp_name,
                                            'time': self._time_start,
                                            'state': self._exp_state,
                                            'current_time': cur_time,
                                            'epoch': self._epoch_idx
                                            }),
                                  "exp")

    def save_temp_checkpoint(self):
        """Saves the current checkpoint as checkpoint_current."""
        self.save_checkpoint(name="checkpoint_current", save_types=self._default_save_types)

    def save_end_checkpoint(self):
        """Saves the current checkpoint as checkpoint_last."""
        self.save_checkpoint(name="checkpoint_last", save_types=self._default_save_types)

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

        r_elem = ResultElement(data=value, label=tag_name, epoch=self._epoch_idx, counter=counter)

        self.results[name] = r_elem

        if plot_result:
            if tag is None:
                legend = False
            else:
                legend = True
            if plot_running_mean:
                value = np.mean(self.results.running_mean_dict[name])
            self.clog.show_value(value=value, name=name, tag=tag_name, counter=counter, show_legend=legend)

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

        if self.elog is None:
            print(*args)
        else:
            self.elog.print(*args)


def get_last_file(dir_, name=None):
    """
    Returns the most recently created file in the folder which matches the name supplied

    Args:
        dir_: The base directory to start the search in
        name: The name pattern to match with the files

    Returns:
        str: the path to the most recent file

    """
    if name is None:
        name = "*checkpoint*.pth.tar"

    dir_files = []

    for root, dirs, files in os.walk(dir_):
        for filename in fnmatch.filter(files, name):
            #if 'last' in filename:
            #    return os.path.join(root, filename)
            checkpoint_file = os.path.join(root, filename)
            dir_files.append(checkpoint_file)

    if len(dir_files) == 0:
        return ""

    last_file = max(dir_files, key=os.path.getctime)

    return last_file


def get_vars_from_sys_argv():
    """
    Parses the command line args (argv) and looks for --config_path and --resume_path and returns them if found.

    Returns:
        tuple: a tuple of (config_path, resume_path ) , None if it is not found

    """
    import sys
    import argparse

    if len(sys.argv) > 1:

        parser = argparse.ArgumentParser()

        # parse just config keys
        parser.add_argument("config_path", type=str)
        parser.add_argument("resume_path", type=str)

        # parse args
        param, unknown = parser.parse_known_args()

        if len(unknown) > 0:
            warnings.warn("Called with unknown arguments: %s" % unknown, RuntimeWarning)

        # update dict
        return param.get("config_path"), param.get("resume_path")


def experimentify(setup_fn="setup", train_fn="train", validate_fn="validate", end_fn="end", test_fn="test", **decoargs):
    """
    Experimental decorator which monkey patches your class into a PytorchExperiment.
    You can then call run on your new :class:`.PytorchExperiment` class.

    Args:
        setup_fn: The name of your setup() function
        train_fn: The name of your train() function
        validate_fn: The name of your validate() function
        end_fn: The name of your end() function
        test_fn: The name of your test() function

    """

    def wrap(cls):

        ### Initilaize both Classes (as original class)
        prev_init = cls.__init__

        def new_init(*args, **kwargs):
            prev_init(*args, **kwargs)
            kwargs.update(decoargs)
            PytorchExperiment.__init__(*args, **kwargs)

        cls.__init__ = new_init

        ### Set new Experiment methods
        if not hasattr(cls, "setup") and hasattr(cls, setup_fn):
            setattr(cls, "setup", getattr(cls, setup_fn))
        elif hasattr(cls, "setup") and setup_fn != "setup":
            warnings.warn("Found already exisiting setup function in class, so will use the exisiting one")

        if not hasattr(cls, "train") and hasattr(cls, train_fn):
            setattr(cls, "train", getattr(cls, train_fn))
        elif hasattr(cls, "train") and setup_fn != "train":
            warnings.warn("Found already exisiting train function in class, so will use the exisiting one")

        if not hasattr(cls, "validate") and hasattr(cls, validate_fn):
            setattr(cls, "validate", getattr(cls, validate_fn))
        elif hasattr(cls, "validate") and setup_fn != "validate":
            warnings.warn("Found already exisiting validate function in class, so will use the exisiting one")

        if not hasattr(cls, "end") and hasattr(cls, end_fn):
            setattr(cls, "end", getattr(cls, end_fn))
        elif hasattr(cls, "end") and end_fn != "end":
            warnings.warn("Found already exisiting end function in class, so will use the exisiting one")

        if not hasattr(cls, "test") and hasattr(cls, test_fn):
            setattr(cls, "test", getattr(cls, test_fn))
        elif hasattr(cls, "test") and test_fn != "test":
            warnings.warn("Found already exisiting test function in class, so will use the exisiting one")

        ### Copy methods from PytorchExperiment into the original class
        for elem in dir(PytorchExperiment):
            if not hasattr(cls, elem):
                trans_fn = getattr(PytorchExperiment, elem)
                setattr(cls, elem, trans_fn)

        return cls

    return wrap
