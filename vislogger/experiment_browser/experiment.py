import json
import numpy as np
import os
import re
from itertools import tee, filterfalse
from vislogger import Config

def partition(pred, iterable):
    t1, t2 = tee(iterable)
    return filter(pred, t1), filterfalse(pred, t2)

class Experiment(object):

    def __init__(self, work_dir, *args, **kwargs):

        super(Experiment, self).__init__(*args, **kwargs)

        self.work_dir = os.path.abspath(work_dir)
        self.config_dir = os.path.join(self.work_dir, "config")
        self.log_dir = os.path.join(self.work_dir, "log")
        self.checkpoint_dir = os.path.join(self.work_dir, "checkpoint")
        self.img_dir = os.path.join(self.work_dir, "img")
        self.plot_dir = os.path.join(self.work_dir, "plot")
        self.save_dir = os.path.join(self.work_dir, "save")
        self.result_dir = os.path.join(self.work_dir, "result")

        self.config = Config()
        self.config.load(os.path.join(self.config_dir, "config.json"))

    def get_file_contents(self, folder):

        if os.path.isdir(folder):
            list_ = map(lambda x: os.path.join(folder, x), sorted(os.listdir(folder)))
            return list(filter(lambda x: os.path.isfile(x), list_))
        else:
            return []

    def get_images(self):
        return self.get_file_contents(self.img_dir)

    def get_plots(self):
        return self.get_file_contents(self.plot_dir)

    def get_checkpoints(self):
        return self.get_file_contents(self.checkpoint_dir)

    def get_logs(self):
        return self.get_file_contents(self.log_dir)

    def get_results(self):

        try:
            n_digits = "{" + str(len(str(self.config.num_epochs))) + "}"
        except:
            n_digits = "+"
        epoch_pattern = ".*\d" + n_digits + ".*\.json"

        result_files = filter(lambda x: re.search(epoch_pattern, x), sorted(os.listdir(self.result_dir)))
        results = {}
        epoch_spacings = {}

        def update_results(new_dict, results_dict, spacings_dict, counter):
            for k in new_dict.keys():
                if isinstance(new_dict[k], dict):
                    flat = {"{}.{}".format(k, key): val for key, val in new_dict[k].items()}
                    update_results(flat, results_dict, spacings_dict, counter)
                else:
                    if k not in results_dict.keys():
                        results_dict[k] = [new_dict[k]]
                        spacings_dict[k] = counter
                    else:
                        results[k].append(new_dict[k])

        for f, file_ in enumerate(result_files):
            with open(os.path.join(self.result_dir, file_), "r") as infile:
                current = json.load(infile)
                update_results(current, results, epoch_spacings, f+1)

        for k in results.keys():
            results[k] = np.array(results[k])

        return results, epoch_spacings
