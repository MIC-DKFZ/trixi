import copy
import json
import os
import time
import warnings
from collections import defaultdict

import numpy as np

from trixi.logger import ExperimentLogger
from trixi.util import Config
from trixi.util.util import StringMultiTypeDecoder


class ExperimentReader(object):
    """Reader class to read out experiments created by :class:`trixi.experimentlogger.ExperimentLogger`.

    Args:
        work_dir (str): Directory with the structure defined by
                        :class:`trixi.experimentlogger.ExperimentLogger`.
        name (str): Optional name for the experiment. If None, will try
                    to read name from experiment config.

    """

    def __init__(self, base_dir, exp_dir="", name=None, decode_config_clean_str=True):

        super(ExperimentReader, self).__init__()

        self.base_dir = base_dir
        self.exp_dir = exp_dir
        self.work_dir = os.path.abspath(os.path.join(self.base_dir, self.exp_dir))
        self.config_dir = os.path.join(self.work_dir, "config")
        self.log_dir = os.path.join(self.work_dir, "log")
        self.checkpoint_dir = os.path.join(self.work_dir, "checkpoint")
        self.img_dir = os.path.join(self.work_dir, "img")
        self.plot_dir = os.path.join(self.work_dir, "plot")
        self.save_dir = os.path.join(self.work_dir, "save")
        self.result_dir = os.path.join(self.work_dir, "result")

        self.config = Config()
        if decode_config_clean_str:
            self.config.load(os.path.join(self.config_dir, "config.json"), decoder_cls_=StringMultiTypeDecoder)
        else:
            self.config.load(os.path.join(self.config_dir, "config.json"), decoder_cls_=None)

        self.exp_info = Config()
        exp_info_file = os.path.join(self.config_dir, "exp.json")
        if os.path.exists(exp_info_file):
            self.exp_info.load(exp_info_file)

        self.__results_dict = None

        self.meta_name = None
        self.meta_star = False
        self.meta_ignore = False
        self.read_meta_info()

        if name is not None:
            self.exp_name = name
        elif self.meta_name is not None:
            self.exp_name = self.meta_name
        elif "name" in self.exp_info:
            self.exp_name = self.exp_info['name']
        elif "exp_name" in self.config:
            self.exp_name = self.config['exp_name']
        else:
            self.exp_name = "experiments"

        self.ignore = self.meta_ignore
        self.star = self.meta_star


    @staticmethod
    def get_file_contents(folder):
        """Get all files in a folder.

        Returns:
            list: All files joined with folder path.
        """

        if os.path.isdir(folder):
            list_ = map(lambda x: os.path.join(folder, x), sorted(os.listdir(folder)))
            return list(filter(lambda x: os.path.isfile(x), list_))
        else:
            return []

    def get_images(self):
        imgs = []
        imgs += ExperimentReader.get_file_contents(self.img_dir)
        if os.path.isdir(self.img_dir):
            for f in os.listdir(self.img_dir):
                f = os.path.join(self.img_dir, f)
                if os.path.isdir(f):
                    imgs += ExperimentReader.get_file_contents(f)
        return imgs

    def get_plots(self):
        return ExperimentReader.get_file_contents(self.plot_dir)

    def get_checkpoints(self):
        return ExperimentReader.get_file_contents(self.checkpoint_dir)

    def get_logs(self):
        return ExperimentReader.get_file_contents(self.log_dir)

    def get_log_file_content(self, file_name):
        """Read out log file and HTMLify.

        Args:
            file_name (str): Name of the log file.

        Returns:
            str: Log file contents as HTML ready string.
        """

        content = ""
        log_file = os.path.join(self.log_dir, file_name)

        if os.path.exists(log_file):
            with open(log_file, 'r') as f:
                content = f.read()
                content = content.replace("\n", "<br>")

        return content

    def get_results_log(self):
        """Build result dictionary.

        During the experiment result items are
        written out as a stream of quasi-atomic units. This reads the stream and
        builds arrays of corresponding items.
        The resulting dict looks like this::

            {
                "result group": {
                    "result": {
                        "counter": x-array,
                        "data": y-array
                    }
                }
            }

        Returns:
            dict: Result dictionary.

        """

        results_merged = {}

        results = []
        try:
            with open(os.path.join(self.result_dir, "results-log.json"), "r") as results_file:
                results = json.load(results_file)
        except Exception as e:
            try:
                with open(os.path.join(self.result_dir, "results-log.json"), "r") as results_file:
                    results_str = results_file.readlines()
                    results_str[-1] = "{}]"
                    results_json = "".join(results_str)
                    results = json.loads(results_json)
            except Exception as ee:
                print("Could not load result log from", self.result_dir)
                print(ee)

        for result in results:
            for key in result.keys():
                counter = result[key]["counter"]
                data = result[key]["data"]
                label = result[key]["label"]
                if label not in results_merged:
                    results_merged[label] = {}
                if key not in results_merged[label]:
                    results_merged[label][key] = {}
                    results_merged[label][key]["data"] = [data]
                    results_merged[label][key]["counter"] = [counter]
                else:
                    results_merged[label][key]["data"].append(data)
                    results_merged[label][key]["counter"].append(counter)

        return results_merged

    def get_results(self):
        """Get the last result item.

        Returns:
            dict: The last result item in the experiment.

        """

        if self.__results_dict is None:

            self.__results_dict = {}
            results_file = os.path.join(self.result_dir, "results.json")

            if os.path.exists(results_file):
                try:
                    with open(results_file, "r") as f:
                        self.__results_dict = json.load(f)
                except Exception as e:
                    pass

        return self.__results_dict

    def ignore_experiment(self):
        """Create a flag file, so the browser ignores this experiment."""
        self.update_meta_info(ignore=True)

    def read_meta_info(self):
        """Reads the meta info of the experiment i.e. new name, stared or ignored"""
        meta_dict = {}
        meta_file = os.path.join(self.work_dir, ".exp_info")
        if os.path.exists(meta_file):
            with open(meta_file, "r") as mf:
                meta_dict = json.load(mf)
            self.meta_name = meta_dict.get("name")
            self.meta_star = meta_dict.get("star", False)
            self.meta_ignore = meta_dict.get("ignore", False)

    def update_meta_info(self, name=None, star=None, ignore=None):
        """
        Updates the meta info i.e. new name, stared or ignored and saves it in the experiment folder

        Args:
            name (str): New name of the experiment
            star (bool): Flag, if experiment is starred/ favorited
            ignore (boll): Flag, if experiment should be ignored
        """

        if name is not None:
            self.meta_name = name
        if star is not None:
            self.meta_star = star
        if ignore is not None:
            self.meta_ignore = ignore

        meta_dict = {
            "name": self.meta_name,
            "star": self.meta_star,
            "ignore": self.meta_ignore
        }
        meta_file = os.path.join(self.work_dir, ".exp_info")
        with open(meta_file, "w") as mf:
            json.dump(meta_dict, mf)


class CombiExperimentReader(ExperimentReader):

    def __init__(self, base_dir, exp_dirs=(), name=None, decode_config_clean_str=True):

        self.base_dir = base_dir

        if name is None or name == "":
            self.exp_name = "combi-experiments"
        else:
            self.exp_name = name

        self.experiments = []
        for exp_dir in exp_dirs:
            self.experiments.append(ExperimentReader(base_dir=base_dir, exp_dir=exp_dir,
                                                     decode_config_clean_str=decode_config_clean_str))

        exp_base_dirs = os.path.commonpath([e.work_dir for e in self.experiments])
        if exp_base_dirs != "":
            self.base_dir = exp_base_dirs

        self.exp_info = Config()

        self.exp_info["epoch"] = -1
        self.exp_info["name"] = self.exp_name
        self.exp_info["state"] = "Combined"
        self.exp_info["time"] = time.strftime("%y-%m-%d_%H:%M:%S", time.localtime(time.time()))

        self.__results_dict = None

        self.work_dir = None
        self.config_dir = "not_saved_yet"
        self.log_dir = "not_saved_yet"
        self.checkpoint_dir = "not_saved_yet"
        self.img_dir = "not_saved_yet"
        self.plot_dir = "not_saved_yet"
        self.save_dir = "not_saved_yet"
        self.result_dir = "not_saved_yet"
        self.exp_dir = "not_saved_yet"

        self.meta_name = None
        self.meta_star = False
        self.meta_ignore = False

        self.config = self.get_config()

        self.elog = None

    def get_config(self):
        combi_config = copy.deepcopy(self.experiments[0].config)
        config_diff = Config.difference_config_static(*[e.config for e in self.experiments])
        combi_config.update(config_diff, deep=True)

        return combi_config

    def get_results_log(self):
        exp_results_logs = [e.get_results_log() for e in self.experiments]

        log_tags = set()
        for res in exp_results_logs:
            log_tags.update(res.keys())

        res_keys = defaultdict(set)
        for tag in log_tags:
            for exp_res in exp_results_logs:
                if tag not in exp_res:
                    continue
                res_keys[tag].update(exp_res[tag].keys())

        combi_res_log = dict()

        for tag, keys in res_keys.items():
            key_result_dict = {}
            for s_key in keys:
                skey_result = defaultdict(list)

                for exp_res in exp_results_logs:
                    if tag not in exp_res:
                        continue
                    if s_key not in exp_res[tag]:
                        continue

                    for c, val in zip(exp_res[tag][s_key]["counter"], exp_res[tag][s_key]["data"]):
                        skey_result[c].append(val)

                key_result_dict[s_key] = skey_result
            combi_res_log[tag] = key_result_dict

        final_results_log = {}

        for tag, key_result_dict in combi_res_log.items():
            final_results_log[tag] = {}
            for s_key, s_key_result_dict in key_result_dict.items():
                final_results_log[tag][s_key] = defaultdict(list)
                cnts =sorted(s_key_result_dict.keys())
                for cnt in cnts:
                    val_list = s_key_result_dict[cnt]
                    final_results_log[tag][s_key]["counter"].append(cnt)
                    final_results_log[tag][s_key]["data"].append(np.median(val_list))
                    final_results_log[tag][s_key]["mean"].append(np.mean(val_list))
                    final_results_log[tag][s_key]["median"].append(np.median(val_list))
                    final_results_log[tag][s_key]["max"].append(np.max(val_list))
                    final_results_log[tag][s_key]["min"].append(np.min(val_list))
                    final_results_log[tag][s_key]["std"].append(np.std(val_list))

        return final_results_log

    def get_results(self):
        exp_results = [e.get_results() for e in self.experiments]
        result_keys = set()
        for res in exp_results:
            result_keys.update(res.keys())

        res_collect = defaultdict(list)

        for key in result_keys:
            for exp_res in exp_results:
                if key not in exp_res:
                    continue
                res_collect[key].append(exp_res[key])

        results_dict = {}
        results_aux_dict = defaultdict(dict)

        for key, val_list in res_collect.items():
            results_dict[key] = np.median(val_list)
            results_aux_dict[key]["mean"] = np.mean(val_list)
            results_aux_dict[key]["median"] = np.median(val_list)
            results_aux_dict[key]["max"] = np.max(val_list)
            results_aux_dict[key]["min"] = np.min(val_list)
            results_aux_dict[key]["std"] = np.std(val_list)

        return results_dict

    def get_result_log_dict(self):

        results_dict = self.get_results_log()

        res_list = []

        for tag, key_result_dict in results_dict.items():
            for s_key, s_key_result_dict in key_result_dict.items():
                for cnt, val in zip(s_key_result_dict["counter"], s_key_result_dict["data"]):
                    res_list.append({s_key:dict(data=val, counter=cnt, epoch=-1, label=tag)})

        return res_list


    def ignore_experiment(self):
        """Create a flag file, so the browser ignores this experiment."""
        if self.work_dir is None:
            warnings.warn("Can only be called for a combined experiment which is saved")
            return
        super(CombiExperimentReader, self).ignore_experiment()

    def read_meta_info(self):
        """Reads the meta info of the experiment i.e. new name, stared or ignored"""
        if self.work_dir is None:
            warnings.warn("Can only be called for a combined experiment which is saved")
            return
        super(CombiExperimentReader, self).read_meta_info()

    def update_meta_info(self, name=None, star=None, ignore=None):        
        if self.work_dir is None:
            warnings.warn("Can only be called for a combined experiment which is saved")
            return
        super(CombiExperimentReader, self).update_meta_info(name, star, ignore)

    def save(self, target_dir=None):

        if target_dir is None:
            target_dir = self.base_dir

        self.elog = ExperimentLogger(experiment_name=self.exp_name, base_dir=target_dir)

        self.exp_dir = self.elog.folder_name
        self.work_dir = self.elog.work_dir
        self.config_dir = os.path.join(self.work_dir, "config")
        self.log_dir = os.path.join(self.work_dir, "log")
        self.checkpoint_dir = os.path.join(self.work_dir, "checkpoint")
        self.img_dir = os.path.join(self.work_dir, "img")
        self.plot_dir = os.path.join(self.work_dir, "plot")
        self.save_dir = os.path.join(self.work_dir, "save")
        self.result_dir = os.path.join(self.work_dir, "result")

        self.config.dump(os.path.join(self.elog.config_dir, "config.json"))
        self.elog.save_config(self.exp_info, "exp")
        self.elog.save_result(self.get_results(), "results", encoder_cls=None)
        self.elog.save_result(self.get_result_log_dict(), "results-log", encoder_cls=None)

        pass

