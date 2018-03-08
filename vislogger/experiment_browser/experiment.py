import os
from vislogger import Config

class Experiment(Config):

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

        self.load(os.path.join(self.config_dir, "config.json"))

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
