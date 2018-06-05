import json
import os

from trixi.util import Config


class ExperimentReader(object):
    """Reader class to read out experiments created by :class:`trixi.experimentlogger.ExperimentLogger`.

    Args:
        work_dir (str): Directory with the structure defined by
                        :class:`trixi.experimentlogger.ExperimentLogger`.
        name (str): Optional name for the experiment. If None, will try
                    to read name from experiment config.

    """

    def __init__(self, base_dir, exp_dir="", name=None):

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
        self.config.load(os.path.join(self.config_dir, "config.json"))

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





