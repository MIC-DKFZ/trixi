import atexit
import fnmatch
import json
import os
import random
import shutil
import string
import time
import traceback
import warnings

import torch

from trixi.experiment.experiment import Experiment
from trixi.logger import CombinedLogger, PytorchExperimentLogger, PytorchVisdomLogger, TelegramLogger
from trixi.util import Config, ResultElement, ResultLogDict, SourcePacker, name_and_iter_to_filename
from trixi.util.pytorchutils import set_seed


class PytorchExperiment(Experiment):
    """
    A Pytorch Experiment is a abstract class which extends the basic functionallity of the Experiment class with
    convenience features for pytorch such as creating a folder structur, saving and plotting results and
    checkpointing your experiment.

    The basic life cycle of a PytorchExperiment is the same a Experiment:

        setup()
        (--> Automatically restore values if a previous checkpoint is given)
        prepare()

        for epoch in n_epochs:
            train()
            validate()
            (--> save current checkpoint)

        end()

    To get your own experiment simply inherit from the PytorchExperiment and overwrite the setup(), prepare(),
    train(), validate() method (or you can use the experimental decorator "experimentify" to convert your
    class into a experiment).
    Then you can run your own experiment by calling the run() method.

    Internally experiment will provide you some member variables which you can access:
        - n_epochs: Number of epochs
        - exp_name: Name of your experiment
        - config: The (initialized) config of your experiment (the serializable/uninitialized config is in _config_raw)
        - result: A dict in which you can store your result values (can and will be persisted if you use a experiment
            logger)
        - vlog (if the flag in the init is True): A visdom logger instance which can log your results to a visdom server
        - elog (flag in init): A experiment logger instance which can log your results to a given folder
        - tlog (flat in init): A telegram logger which can send the results to your telegram account
        - clog: A combined logger which logs to all loggers in different frequencies (which can be defined)

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
                 parse_sys_argv=False,
                 parse_config_sys_argv=True,
                 checkpoint_to_cpu=True,
                 use_visdomlogger=True,
                 visdomlogger_kwargs=None,
                 visdomlogger_c_freq=1,
                 use_explogger=True,
                 explogger_kwargs=None,
                 explogger_c_freq=100,
                 use_telegramlogger=False,
                 telegramlogger_kwargs=None,
                 telegramlogger_c_freq=1000,
                 append_rnd_to_name=False):
        """
        Initializes the Pytorch experiment and creates the basic experiment infrastructure

        Args:
            config (dict or Config): A config, if name, n_epochs, seed, base_dir, given in the config it will
            automatically overwrite the other args/kwargs with the values from the config. In addition (defined by
            parse_config_sys_argv) the config automatically parses the argv arguments and updates its values if a
             key matches a console argument
            name (str): The name of the PytorchExperiment
            n_epochs (int): The number of epochs (number of times the training cycle will be executed)
            seed (int): A random seed (which will set the random, numpy and torch seed)
            base_dir (str): A base directory in which the experiment result folder will be created
            globs: the globals() of the script which in run. This is nesseary to get and save the executed files in
                the experiment folder.
            resume (str of PytorchExperiment): Another Pytorch experiment or path to the result dir from another
                PytorchExperiment from which in will load the pytorch modules and other member variables and resume
                the experiment
            ignore_resume_config (bool): If True it will not load resume with the config from the resume Experiment
                but take the current/own config
            resume_save_types (list): A list which can define which values to restore when resuming. Choices are:
                ("model" <-- Pytorch models, "optimizer" <-- Optimizers, "simple" <-- Simple python variables (basic
                types and list/tuples ), "th_vars" <-- torch tensors/variables, "results" <-- The result dict)
            parse_sys_argv (bool): Parsing the console arguments (argv) to get a config_path and/or resume_path
            parse_config_sys_argv (bool): Parse argv to update the config (if the keys match)
            checkpoint_to_cpu (bool): When checkpointing transfer all tensors to the cpu beforehand
            use_visdomlogger (bool): Use a pytorch visdom logger. Is accessible via the vlog variable
            visdomlogger_kwargs (dict): Keyword arguments will are passed to the pytorch visdom logger initialization
            visdomlogger_c_freq (int): The frequency x ( == one in x) in which the combined logger will call the visdom
                logger
            use_explogger (bool): Use a experiment logger. Is accessible via the elog variable. It will create the
                experiment folder structure
            explogger_kwargs (dict): Keyword arguments will are passed to the experiment logger initialization
            explogger_c_freq (int): The frequency in which the combined logger will call the experiment logger
            use_telegramlogger (bool): Use a telegram logger. Is accessible via the tlog variable.
            telegramlogger_kwargs (dict):  Keyword arguments will are passed to the telegram logger initialization
            telegramlogger_c_freq (int): The frequency in which the combined logger will call the telegram logger
            append_rnd_to_name (bool): If True will append a random six digit string to the experiment name
        """
        # super(PytorchExperiment, self).__init__()
        Experiment.__init__(self)

        if parse_sys_argv:
            config_path, resume_path = get_vars_from_sys_argv()
            if config_path:
                config = config_path
            if resume_path:
                resume = resume_path

        self._config_raw = None
        if isinstance(config, str):
            self._config_raw = Config(file_=config, update_from_argv=parse_config_sys_argv)
        elif isinstance(config, Config):
            self._config_raw = Config(config=config, update_from_argv=parse_config_sys_argv)
        elif isinstance(config, dict):
            self._config_raw = Config(config=config, update_from_argv=parse_config_sys_argv)
        else:
            self._config_raw = Config(update_from_argv=parse_config_sys_argv)

        self.n_epochs = n_epochs
        if 'n_epochs' in self._config_raw:
            self.n_epochs = self._config_raw["n_epochs"]

        self._seed = seed
        if 'seed' in self._config_raw:
            self._seed = self._config_raw.seed
        if self._seed is None:
            random_data = os.urandom(4)
            seed = int.from_bytes(random_data, byteorder="big")
            self._config_raw.seed = seed
            self._seed = seed

        self.exp_name = name
        if 'name' in self._config_raw:
            self.exp_name = self._config_raw["name"]
        if append_rnd_to_name:
            rnd_str = ''.join(random.choice(string.ascii_letters + string.digits) for _ in range(5))
            self.exp_name += "-" + rnd_str

        if 'base_dir' in self._config_raw:
            base_dir = self._config_raw["base_dir"]

        self._checkpoint_to_cpu = checkpoint_to_cpu
        self.results = dict()

        # Init loggers
        logger_list = []
        if use_visdomlogger:
            if visdomlogger_kwargs is None:
                visdomlogger_kwargs = {}
            self.vlog = PytorchVisdomLogger(name=self.exp_name, **visdomlogger_kwargs)
            if visdomlogger_c_freq is not None and visdomlogger_c_freq > 0:
                logger_list.append((self.vlog, visdomlogger_c_freq))
        if use_explogger:
            if explogger_kwargs is None:
                explogger_kwargs = {}
            self.elog = PytorchExperimentLogger(base_dir=base_dir,
                                                experiment_name=self.exp_name,
                                                **explogger_kwargs)
            if explogger_c_freq is not None and explogger_c_freq > 0:
                logger_list.append((self.elog, explogger_c_freq))

            # Set results log dict to the right path
            self.results = ResultLogDict("results-log.json", base_dir=self.elog.result_dir)
            self.results.print_to_file("[")
        if use_telegramlogger:
            if telegramlogger_kwargs is None:
                telegramlogger_kwargs = {}
            self.tlog = TelegramLogger(**telegramlogger_kwargs, exp_name=self.exp_name)
            if telegramlogger_c_freq is not None and telegramlogger_c_freq > 0:
                logger_list.append((self.tlog, telegramlogger_c_freq))

        self.clog = CombinedLogger(*logger_list)

        set_seed(self._seed)

        # Do the resume stuff
        self._resume_path = None
        self._resume_save_types = resume_save_types
        self._ignore_resume_config = ignore_resume_config
        if resume is not None:
            if isinstance(resume, str):
                self._resume_path = resume
            elif isinstance(resume, PytorchExperiment):
                self._resume_path = resume.elog.base_dir

        # self.elog.save_config(self.config, "config_pre")
        if globs is not None:
            zip_name = os.path.join(self.elog.save_dir, "sources.zip")
            SourcePacker.zip_sources(globs, zip_name)

        # Init objects in config
        self.config = Config.init_objects(self._config_raw)

        atexit.register(self.at_exit_func)

    def process_err(self, e):
        self.elog.text_logger.log_to("\n".join(traceback.format_tb(e.__traceback__)), "err")

    def update_attributes(self, var_dict, ignore=()):
        """
        Updates the member attributes with the attributes given in the var_dict

        Args:
            var_dict: dict in which the values which are updated are stored. If a key matches a member attribute name
                the member attribute will be updated
            ignore: a list of keys to ignore

        """
        for key, val in var_dict.items():
            if key in ignore:
                continue
            if hasattr(self, key):
                setattr(self, key, val)

    def get_pytorch_modules(self):
        """Returns all pytorch (nn) modules stored in the algo in a dict"""
        pyth_modules = dict()
        for key, val in self.__dict__.items():
            if isinstance(val, torch.nn.Module):
                pyth_modules[key] = val
        return pyth_modules

    def get_pytorch_optimizers(self):
        """Returns all pytorch optimizers stored in the algo in a dict"""
        pyth_optimizers = dict()
        for key, val in self.__dict__.items():
            if isinstance(val, torch.optim.Optimizer):
                pyth_optimizers[key] = val
        return pyth_optimizers

    def get_simple_variables(self, ignore=()):
        """
        Returns all variables in the experiment which might be interesting in a dict.

        Args:
            ignore: A list of names, which will be ignores
        """
        simple_vars = dict()
        for key, val in self.__dict__.items():
            if key in ignore:
                continue
            if isinstance(val, (int, float, bytes, bool, str, set, list, tuple)):
                simple_vars[key] = val
        return simple_vars

    def get_pytorch_variables(self, ignore=()):
        """Returns all variables in the experiment which might be interesting"""
        pytorch_vars = dict()
        for key, val in self.__dict__.items():
            if key in ignore:
                continue
            if isinstance(val, (*torch._tensor_classes, torch.autograd.Variable)):
                pytorch_vars[key] = val
        return pytorch_vars

    def save_results(self, name="results.json"):
        """
        Saves the result dict as a json file in the result dir of the experiment logger.

        Args:
            name: The name of the json file, in which the results are written.

        """
        with open(os.path.join(self.elog.result_dir, name), "w") as file_:
            json.dump(self.results, file_, indent=4)

    def save_pytorch_models(self):
        """Saves all pytorch models as model files in the experiment model folder"""
        pyth_modules = self.get_pytorch_modules()
        for key, val in pyth_modules.items():
            self.elog.save_model(val, key)

    def load_pytorch_models(self):
        """Loads all pytorch models as models frpom the files in the experiment model folder"""
        pyth_modules = self.get_pytorch_modules()
        for key, val in pyth_modules.items():
            self.elog.load_model(val, key)

    def log_simple_vars(self):
        """Logs all simple python member variables as a json file in the log dir"""
        simple_vars = self.get_simple_variables()
        with open(os.path.join(self.elog.log_dir, "simple_vars.json"), "w") as file_:
            json.dump(simple_vars, file_)

    def load_simple_vars(self):
        """Restores all simple python member variables from a json file in the log dir"""
        simple_vars = {}
        with open(os.path.join(self.elog.log_dir, "simple_vars.json"), "r") as file_:
            simple_vars = json.load(file_)
        self.update_attributes(simple_vars)

    def save_checkpoint(self, name="checkpoint", save_types=("model", "optimizer", "simple", "th_vars", "results"),
                        n_iter=None, iter_format="{:05d}", prefix=False):
        """
        Saves a current checkpoint from the experiment.

        Args:
            name: The name of the checkpoint file
            save_types: What kind of member variables should be stored: Choices are:
                ("model" <-- Pytorch models, "optimizer" <-- Optimizers, "simple" <-- Simple python variables (basic
                types and list/tuples ), "th_vars" <-- torch tensors/variables, "results" <-- The result dict)
            n_iter: Number of iteration. together with the name, defined by the iter_format a file name will be created
            iter_format: Defines how the name and the n_iter will be combined
            prefix: If True, the formated n_iter will be appended as a prefix, otherwise as a suffix

        """

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

    def load_checkpoint(self, name="checkpoint", save_types=("model", "optimizer", "simple", "th_vars", "results"),
                        n_iter=None, iter_format="{:05d}", prefix=False, path=None):

        """
        Loads a checkpoint and restores the experiment.

        Args:
            name: The name of the checkpoint file
            save_types: What kind of member variables should be stored: Choices are:
                ("model" <-- Pytorch models, "optimizer" <-- Optimizers, "simple" <-- Simple python variables (basic
                types and list/tuples ), "th_vars" <-- torch tensors/variables, "results" <-- The result dict)
            n_iter: Number of iteration. together with the name, defined by the iter_format a file name will be created
            iter_format: Defines how the name and the n_iter will be combined
            prefix: If True, the formated n_iter will be appended as a prefix, otherwise as a suffix
            path: If a path is given than it will take the current experiment dir and formated name, otherwise it will
                simple use the path and the formatted name to define the checkpoint file

        """

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
            if checkpoint_path.endswith("/"):
                checkpoint_path = checkpoint_path[:-1]
            restore_dict = self.elog.load_checkpoint_static(checkpoint_file=checkpoint_path, **checkpoint_dict)

        self.update_attributes(restore_dict)

    def end(self):
        """Ends the experiment and stores the final results/ checkpoint"""
        if isinstance(self.results, ResultLogDict):
            self.results.print_to_file("]")
        self.save_results()
        self.save_end_checkpoint()
        self.elog.save_config(Config(**{'name': self.exp_name, 'time': self._time_start, 'state': self._exp_state}),
                              "exp")
        self.elog.print("Experiment ended. Checkpoints stored =)")

    def end_test(self):
        """Ends the experiment and stores the final results and config"""
        self.save_results()
        self.elog.save_config(Config(**{'name': self.exp_name, 'time': self._time_start, 'state': self._exp_state}),
                              "exp")
        self.elog.print("Testing ended. Results stored =)")

    def at_exit_func(self):
        """Stores the results and checkpoint at the end (if nor already stored). This method is also called if an
        error occurs"""
        if self._exp_state not in ("Ended", "Tested"):
            if isinstance(self.results, ResultLogDict):
                self.results.print_to_file("]")
            self.save_checkpoint(name="checkpoint_exit-" + self._exp_state)
            self.save_results()
            self.elog.save_config(Config(**{'name': self.exp_name, 'time': self._time_start, 'state': self._exp_state}),
                                  "exp")
            self.elog.print("Experiment exited. Checkpoints stored =)")
        time.sleep(10)  # allow checkpoint saving to finish

    def _setup_internal(self):
        self.prepare_resume()
        self.elog.save_config(self._config_raw, "config")
        self.elog.save_config(Config(**{'name': self.exp_name, 'time': self._time_start, 'state': self._exp_state}),
                              "exp")

    def _start_internal(self):
        self.elog.save_config(Config(**{'name': self.exp_name, 'time': self._time_start, 'state': self._exp_state}),
                              "exp")

    def prepare_resume(self):
        """Tries to resume the experiment by using the defined resume path or resume PytorchExperiment"""
        checkpoint_file = ""
        base_dir = ""

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
                self.elog.print("Loaded existing config from:", base_dir)

        if checkpoint_file:
            self.load_checkpoint(name="", path=checkpoint_file, save_types=self._resume_save_types)
            self._resume_path = checkpoint_file
            shutil.copyfile(checkpoint_file, os.path.join(self.elog.checkpoint_dir, "0_checkpoint.pth.tar"))
            self.elog.print("Loaded existing checkpoint from:", checkpoint_file)

    def _end_epoch_internal(self, epoch):
        self.save_results()
        self.save_temp_checkpoint()

    def save_temp_checkpoint(self):
        """Saves the current checkpoint as checkpoint_current"""
        self.save_checkpoint(name="checkpoint_current")

    def save_end_checkpoint(self):
        """Saves the current checkpoint as checkpoint_last"""
        self.save_checkpoint(name="checkpoint_last")

    def add_result(self, value, name, counter=None, label=None, plot_result=True):
        """
        Saves a results and add it to the result dict, this is similar to results[key] = val, but in addition also
        logs the value to the combined logger (it also stores in the results-logs file).

        **This should be your preferred method to log your numeric values**

        Args:
            value: The value of your variable
            name: The name/ key of your variable
            counter: A counter which can be seen a the x-axis of your value
            label: A label/ tag which can group similar values and will plot values with the same label in the same plot
            plot_result: By default True, will also log all your values to the combined logger (with show value)

        """

        label_name = label
        if label_name is None:
            label_name = name

        r_elem = ResultElement(data=value, label=label_name, epoch=self._epoch_idx, counter=counter)

        self.results[name] = r_elem

        if plot_result:
            if label is None:
                plt_name = name
                tag = None
                legend = False
            else:
                plt_name = label
                tag = name
                legend = True
            self.clog.show_value(value=value, name=plt_name, tag=tag, counter=counter, show_legend=legend)

    def get_result(self, name):
        """
        Similar to result[key] this will return the values in result with the given name/key

        Args:
            name: the name/ key for which a value is stores

        Returns: The value with the key 'name' in the results dict

        """
        return self.results.get(name)

    def add_result_without_epoch(self, val, name):
        """
        A faster method to store your results, has less overhead and does not call the combined logger

        Args:
            val: the values you want to add
            name: the name/ key of your value

        """
        self.results[name] = val

    def get_result_without_epoch(self, name):
        """
        Similar to result[key] this will return the values in result with the given name/key

        Args:
            name: the name/ key for which a value is stores

        Returns: The value with the key 'name' in the results dict

        """
        return self.results.get(name)


def get_last_file(dir_, name=None):
    """
    Returns the (alphabetically) last file in the folder which matches the name supplied

    Args:
        dir_: The base directory to start the search in
        name: The name pattern to match with the files

    Returns: the path to the (alphabetically) last file

    """
    if name is None:
        name = "*checkpoint*.pth.tar"

    dir_files = []

    for root, dirs, files in os.walk(dir_):
        for filename in fnmatch.filter(files, name):
            if 'last' in filename:
                return os.path.join(root, filename)
            checkpoint_file = os.path.join(root, filename)
            dir_files.append(checkpoint_file)

    if len(dir_files) == 0:
        return ""

    last_file = sorted(dir_files, reverse=True)[0]

    return last_file


def get_vars_from_sys_argv():
    """
    Parses the command line args (argv) and looks for --config_path and --resume_path and returns them if found.

    Returns: a Tuple of (config_path, resume_path ) , None if it is not found

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
    Experimental decorator with monkey patches your class into a PytorchExperiment.
    You can then call run on your new PytorchExperiment Class.

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
