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

        with open(os.path.join(self.result_dir, "results-log.json"), "r") as results_file:
            results = json.load(results_file)
        results_merged = {}

        for result in results:
            for key in result.keys():
                counter = result[key]["counter"]
                data = result[key]["data"]
                epoch = result[key]["epoch"]
                label = result[key]["label"]
                if label not in results_merged:
                    results_merged[label] = {}
                if key not in results_merged[label]:
                    results_merged[label][key] = {}
                    results_merged[label][key]["data"] = [data]
                    results_merged[label][key]["epoch"] = [epoch]
                else:
                    if counter < len(results_merged[label][key]["data"]):
                        raise IndexError("Tried to insert element with counter {} into {}.{}.data, but there are already {} elements.".format(
                            counter, label, key, results_merged[label][key]["data"]))
                    else:
                        results_merged[label][key]["data"].append(data)
                        results_merged[label][key]["epoch"].append(epoch)

        return results_merged
