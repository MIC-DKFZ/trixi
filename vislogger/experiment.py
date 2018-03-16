import atexit
import fnmatch
import json
import os.path
import random
import shutil
import time
import traceback
import warnings

import numpy as np
import torch
import vislogger
from vislogger import Config
from vislogger.sourcepacker import SourcePacker
from vislogger.util import name_and_iter_to_filename


class Experiment(object):
    def __init__(self, n_epochs=0):
        # super(Experiment, self).__init__()

        self.n_epochs = n_epochs
        self.exp_state = "Preparing"
        self.time_start = ""
        self.time_end = ""

    def run(self):
        """This method runs the experiment"""

        try:
            self.time_start = time.strftime("%y-%m-%d_%H:%M:%S", time.localtime(time.time()))
            self.time_end = ""

            self.setup()
            self._setup_internal()

            self.exp_state = "Started"
            for epoch in range(self.n_epochs):
                self.train(epoch=epoch)
                self.validate(epoch=epoch)
                self._end_epoch_internal(epoch=epoch)

            self.exp_state = "Trained"

            print("Trained.")
            self.end()
            self.exp_state = "Ended"

            self.time_end = time.strftime("%y-%m-%d_%H:%M:%S", time.localtime(time.time()))

        except Exception as e:

            # run_error = e
            # run_error_traceback = traceback.format_tb(e.__traceback__)
            self.exp_state = "Error"
            self.process_err(e)
            self.time_end = time.strftime("%y-%m-%d_%H:%M:%S", time.localtime(time.time()))

            raise e

    def run_test(self, setup=True):
        """This method runs the experiment"""

        try:

            if setup:
                self.setup()
                self._setup_internal()

            self.exp_state = "Testing"
            self.test()
            self.end_test()
            self.exp_state = "Tested"

            print("Tested.")

        except Exception as e:

            # run_error = e
            # run_error_traceback = traceback.format_tb(e.__traceback__)
            self.exp_state = "Error"
            self.process_err(e)

            raise e

    def setup(self):
        """Is called at the beginning of each experiment run to setup the basic components needed for a run"""
        pass

    def train(self, epoch):
        """The training part of the experiment, it is called once for each epoch"""
        pass

    def validate(self, epoch):
        """The evaluation/valdiation part of the experiment, it is called once for each epoch (after the training
        part)"""
        pass

    def test(self):
        """The testing part of the experiment"""
        pass

    def process_err(self, e):
        pass

    def _setup_internal(self):
        pass

    def _end_epoch_internal(self, epoch):
        pass

    def end(self):
        """Is called at the end of each experiment"""
        pass

    def end_test(self):
        """Is called at the end of each experiment test"""
        pass


class PyTorchExperiment(Experiment):
    def __init__(self, config=None, name=None, n_epochs=None, seed=None, base_dir=None, globs=None, resume=None,
                 ignore_resume_config=False, resume_save_types=("model", "optimizer", "simple", "th_vars", "results"),
                 parse_sys_argv=False):
        """Inits an algo with a config, config needs to a n_epochs, name, output_folder and seed !"""
        # super(PyTorchExperiment, self).__init__()
        Experiment.__init__(self)

        if parse_sys_argv:
            config_path, resume_path = get_vars_from_sys_argv()
            if config_path:
                config = config_path
            if resume_path:
                resume = resume_path

        self.__config_raw = None
        if isinstance(config, str):
            self.__config_raw = Config(file_=config, update_from_argv=True)
        elif isinstance(config, Config):
            self.__config_raw = config
        elif isinstance(config, dict):
            self.__config_raw = Config(config=config)
        else:
            self.__config_raw = Config(update_from_argv=True)

        self.n_epochs = n_epochs
        if "n_epochs" in config:
            self.n_epochs = config.n_epochs

        self.seed = seed
        if "seed" in config:
            self.seed = config.seed
        if seed is None:
            random_data = os.urandom(4)
            seed = int.from_bytes(random_data, byteorder="big")
            config.seed = seed
            self.seed = seed

        self.exp_name = name
        if "name" in config:
            name = config.name
            self.exp_name = config.name

        if "base_dir" in config:
            base_dir = config.base_dir

        self.vlog = vislogger.pytorchvisdomlogger.PytorchVisdomLogger(name=self.exp_name)
        self.elog = vislogger.pytorchexperimentlogger.PytorchExperimentLogger(base_dir=base_dir, experiment_name=name)
        self.clog = vislogger.CombinedLogger((self.vlog, 1), (self.elog, 100))

        set_seed(self.seed)

        self.results = dict()

        self.resume_path = None
        self.resume_save_types = resume_save_types
        self.ignore_resume_config = ignore_resume_config
        if resume is not None:
            if isinstance(resume, str):
                self.resume_path = resume
            elif isinstance(resume, PyTorchExperiment):
                self.resume_path = resume.elog.base_dir

        # self.elog.save_config(self.config, "config_pre")
        if globs is not None:
            zip_name = os.path.join(self.elog.save_dir, "sources.zip")
            SourcePacker.zip_sources(globs, zip_name)

        # Init objects in config
        self.config = Config.init_objects(self.__config_raw)

        atexit.register(self.at_exit_func)

    def process_err(self, e):
        self.elog.text_logger.log_to("\n".join(traceback.format_tb(e.__traceback__)), "err")
        print("err", "\n".join(traceback.format_tb(e.__traceback__)))

    def update_attributes(self, var_dict, ignore=()):
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
        """Returns all variables in the experiment which might be interesting"""
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
        with open(os.path.join(self.elog.result_dir, name), "w") as file_:
            json.dump(self.results, file_)

    def save_pytorch_models(self):
        pyth_modules = self.get_pytorch_modules()
        for key, val in pyth_modules.items():
            self.elog.save_model(val, key)

    def load_pytorch_models(self):
        pyth_modules = self.get_pytorch_modules()
        for key, val in pyth_modules.items():
            self.elog.load_model(val, key)

    def log_simple_vars(self):
        simple_vars = self.get_simple_variables()
        with open(os.path.join(self.elog.log_dir, "simple_vars.log"), "w") as file_:
            json.dump(simple_vars, file_)

    def load_simple_vars(self):
        simple_vars = {}
        with open(os.path.join(self.elog.log_dir, "simple_vars.log"), "r") as file_:
            simple_vars = json.load(file_)
        self.update_attributes(simple_vars)

    def save_checkpoint(self, name="checkpoint", save_types=("model", "optimizer", "simple", "th_vars", "results"),
                        n_iter=None, iter_format="{:05d}", prefix=False):

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

        self.elog.save_checkpoint(name=name, n_iter=n_iter, iter_format=iter_format, prefix=prefix, **checkpoint_dict)

    def load_checkpoint(self, name="checkpoint", save_types=("model", "optimizer", "simple", "th_vars", "results"),
                        n_iter=None, iter_format="{:05d}", prefix=False, path=None):

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
        self.save_results()
        self.save_end_checkpoint()
        self.elog.print("Experiment ended. Checkpoints stored =)")

    def end_test(self):
        self.save_results(name="results-test")
        self.elog.print("Testing ended. Results stored =)")

    def at_exit_func(self):
        if self.exp_state not in ("Ended", "Tested"):
            self.save_results(name="results-" + self.exp_state + ".log")
            self.save_checkpoint(name="checkpoint_exit-" + self.exp_state)
            self.elog.print("Experiment exited. Checkpoints stored =)")

    def _setup_internal(self):
        self.prepare_resume()
        self.elog.save_config(self.__config_raw, "config")

    def prepare_resume(self):
        checkpoint_file = ""
        base_dir = ""

        if self.resume_path is not None:
            if isinstance(self.resume_path, str):
                if self.resume_path.endswith(".pth.tar"):
                    checkpoint_file = self.resume_path
                    base_dir = os.path.dirname(os.path.dirname(checkpoint_file))
                elif self.resume_path.endswith("checkpoint") or self.resume_path.endswith("checkpoint/"):
                    checkpoint_file = get_last_file(self.resume_path)
                    base_dir = os.path.dirname(os.path.dirname(checkpoint_file))
                elif "checkpoint" in os.listdir(self.resume_path) and "config" in os.listdir(self.resume_path):
                    checkpoint_file = get_last_file(self.resume_path)
                    base_dir = self.resume_path
                else:
                    warnings.warn("You have not selected a valid experiment folder, will search all sub folders",
                                  UserWarning)
                    self.elog.text_logger.log_to("You have not selected a valid experiment folder, will search all "
                                                 "sub folders", "warnings")
                    checkpoint_file = get_last_file(self.resume_path)
                    base_dir = os.path.dirname(os.path.dirname(checkpoint_file))

        if base_dir:
            if not self.ignore_resume_config:
                load_config = Config()
                load_config.load(os.path.join(base_dir, "config/config.json"))
                self.__config_raw = load_config
                self.config = Config.init_objects(self.__config_raw)
                self.elog.print("Loaded existing config from:", base_dir)

        if checkpoint_file:
            self.load_checkpoint(name="", path=checkpoint_file, save_types=self.resume_save_types)
            self.resume_path = checkpoint_file
            shutil.copyfile(checkpoint_file, os.path.join(self.elog.checkpoint_dir, "0_checkpoint.pth.tar"))
            self.elog.print("Loaded existing checkpoint from:", checkpoint_file)

    def _end_epoch_internal(self, epoch):
        self.save_temp_checkpoint()

    def save_temp_checkpoint(self):
        self.save_checkpoint(name="checkpoint_current")

    def save_end_checkpoint(self):
        self.save_checkpoint(name="checkpoint_last")


def get_last_file(dir_, name=None):
    if name is None:
        name = "*checkpoint*.pth.tar"

    dir_files = []

    for root, dirs, files in os.walk(dir_):
        for filename in fnmatch.filter(files, name):
            checkpoint_file = os.path.join(root, filename)
            dir_files.append(checkpoint_file)

    if len(dir_files) == 0:
        return ""

    last_file = sorted(dir_files, reverse=True)[0]

    return last_file


def get_vars_from_sys_argv():
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


def set_seed(seed):
    """Sets the seed"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def experimentify(setup_fn="setup", train_fn="train", validate_fn="validate", end_fn="end", test_fn="test", **decoargs):
    def wrap(cls):

        ### Initilaize both Classes (as original class)
        prev_init = cls.__init__

        def new_init(*args, **kwargs):
            prev_init(*args, **kwargs)
            kwargs.update(decoargs)
            PyTorchExperiment.__init__(*args, **kwargs)

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

        ### Copy methods from PyTorchExperiment into the original class
        for elem in dir(PyTorchExperiment):
            if not hasattr(cls, elem):
                trans_fn = getattr(PyTorchExperiment, elem)
                setattr(cls, elem, trans_fn)

        return cls

    return wrap
