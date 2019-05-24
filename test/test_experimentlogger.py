import os
import tempfile
import unittest
import shutil
import time

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from imageio import imread

from trixi.logger.experiment.experimentlogger import ExperimentLogger
from trixi.util.config import Config

test_dir = "test_dir"


class TestExperimentLogger(unittest.TestCase):

    def setUp(self):
        self.test_dir = tempfile.gettempdir()
        self.experimentLogger = ExperimentLogger(exp_name="test",
                                                 base_dir=self.test_dir,
                                                 folder_format="{experiment_name}")

    def tearDown(self):
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_create_folders(self):
        exp_dir = os.path.join(self.test_dir, "test")
        self.assertTrue(os.path.exists(exp_dir), "Experiment dir not created")
        self.assertTrue(os.path.exists(os.path.join(exp_dir, "checkpoint")), "Config dir not created")
        self.assertTrue(os.path.exists(os.path.join(exp_dir, "config")), "Config dir not created")
        self.assertTrue(os.path.exists(os.path.join(exp_dir, "img")), "Config dir not created")
        self.assertTrue(os.path.exists(os.path.join(exp_dir, "log")), "Config dir not created")
        self.assertTrue(os.path.exists(os.path.join(exp_dir, "config")), "Config dir not created")
        self.assertTrue(os.path.exists(os.path.join(exp_dir, "plot")), "Config dir not created")
        self.assertTrue(os.path.exists(os.path.join(exp_dir, "result")), "Config dir not created")
        self.assertTrue(os.path.exists(os.path.join(exp_dir, "save")), "Config dir not created")

    def test_two_experiment_loggers_same_test_dir_no_run_number_throws_error(self):
        logger = ExperimentLogger(exp_name="test",
                                  base_dir=self.test_dir,
                                  folder_format="{experiment_name}")
        self.assertTrue(os.path.isdir(logger.base_dir), "Experiment directory not created")

    def test_show_image(self):
        image = np.random.random_sample((3, 128, 128))
        self.experimentLogger.show_image(image, "image")
        self.assertTrue(os.path.exists(os.path.join(self.experimentLogger.img_dir, "image.png")),
                        "Show image could not create image")

    def test_show_barplot(self):
        array = np.random.random_sample(5)
        self.experimentLogger.show_barplot(array, "barplot")
        self.assertTrue(os.path.exists(os.path.join(self.experimentLogger.plot_dir, "barplot.png")),
                        "Show barplot could not create barplot")

    def test_show_lineplot(self):
        x = [0, 1, 2, 3, 4, 5]
        y = np.random.random_sample(6)
        self.experimentLogger.show_lineplot(y, x, name="lineplot1")
        self.assertTrue(os.path.exists(os.path.join(self.experimentLogger.plot_dir, "lineplot1.png")),
                        "Show lineplot could not create lineplot")
        self.experimentLogger.show_lineplot(y, name="lineplot2")
        self.assertTrue(os.path.exists(os.path.join(self.experimentLogger.plot_dir, "lineplot2.png")),
                        "Show lineplot could not create lineplot without x vals")

    def test_show_piechart(self):
        array = np.random.random_sample(5)
        self.experimentLogger.show_piechart(array, "piechart")
        self.assertTrue(os.path.exists(os.path.join(self.experimentLogger.plot_dir, "piechart.png")),
                        "Show piechart could not create piechart")

    def test_show_scatterplot(self):
        array = np.random.random_sample((5, 2))
        self.experimentLogger.show_scatterplot(array, "scatterplot")
        self.assertTrue(os.path.exists(os.path.join(self.experimentLogger.plot_dir, "scatterplot.png")),
                        "Show scatterplot could not create scatterplot")

    def test_show_value(self):
        val = np.random.random_sample(1)
        self.experimentLogger.show_value(val, "value")
        plt1_content = imread(os.path.join(self.experimentLogger.plot_dir, "value.png"))

        val = np.random.random_sample(1)
        self.experimentLogger.show_value(val, "value")
        plt2_content = imread(os.path.join(self.experimentLogger.plot_dir, "value.png"))
        self.assertFalse(np.array_equal(plt1_content, plt2_content), "Show value did not update the plot")

        val = np.random.random_sample(1)
        self.experimentLogger.show_value(val, "value", counter=4)
        plt3_content = imread(os.path.join(self.experimentLogger.plot_dir, "value.png"))
        self.assertFalse(np.array_equal(plt2_content, plt3_content), "Show value did not update the plot")

    def test_show_text(self):
        text = "\nTest 4 fun: zD ;-D 0o"
        self.experimentLogger.show_text(text)
        log_text = ""
        with open(os.path.join(self.experimentLogger.log_dir, "default.log"), 'r') as log_file:
            log_text = log_file.read()
        self.assertTrue(text in log_text)



    def test_save_and_load_config(self):
        c = Config()
        c.text = "0o"
        c.nmbr = 4
        c.cls = str
        c.lst = [1, 2, 3, 4]

        self.experimentLogger.save_config(c, "config")
        config_path = os.path.join(self.experimentLogger.config_dir, "config.json")
        self.assertTrue(os.path.exists(config_path),
                        "Config could not be saved")

        c2 = self.experimentLogger.load_config("config")
        self.assertTrue("text" in c2 and c2.text == c.text, "Text in config could not be restored")
        self.assertTrue("nmbr" in c2 and c2.nmbr == c.nmbr, "Number in config could not be restored")
        self.assertTrue("cls" in c2 and c2.cls == c.cls, "Text in config could not be restored")
        self.assertTrue("lst" in c2 and c2.lst == c.lst, "List in config could not be restored")

    def test_save_results(self):
        d = dict()
        d['best_val'] = 3.1415926
        d['worst_val'] = 2.718281

        self.experimentLogger.save_result(d, "result")
        result_path = os.path.join(self.experimentLogger.result_dir, "result.json")
        self.assertTrue(os.path.exists(result_path),
                        "Result could not be saved")
        result_text = ""
        with open(result_path, 'r') as log_file:
            result_text = log_file.read()
        self.assertTrue("best_val" in result_text, "Saved vals could not be found in result")
        self.assertTrue("3.1415926" in result_text, "Saved vals could not be found in result")
        self.assertTrue("worst_val" in result_text, "Saved vals could not be found in result")
        self.assertTrue("2.718281" in result_text, "Saved vals could not be found in result")

    def test_save_and_load_dict(self):
        d = dict()
        d['text'] = "0o"
        d['nmbr'] = 4
        d['tuple'] = (1, 2, 3, 4)

        self.experimentLogger.save_dict(d, "dict")
        dict_path = os.path.join(self.experimentLogger.save_dir, "dict.json")
        self.assertTrue(os.path.exists(dict_path),
                        "dict could not be saved")

        d2 = self.experimentLogger.load_dict("dict")
        self.assertTrue("text" in d2 and d2['text'] == d['text'], "Text in dict could not be restored")
        self.assertTrue("nmbr" in d2 and d2['nmbr'] == d['nmbr'], "Number in dict could not be restored")
        self.assertTrue("tuple" in d2 and d2['tuple'] == d['tuple'], "Tuple in dict could not be restored")


    def test_save_and_load_numpy_data(self):
        np_array = np.random.random_sample((3, 128, 128))

        self.experimentLogger.save_numpy_data(np_array, "array")
        np_path = os.path.join(self.experimentLogger.save_dir, "array.npy")
        self.assertTrue(os.path.exists(np_path),
                        "Np Array could not be saved")

        array2 = self.experimentLogger.load_numpy_data("array")
        self.assertTrue(np.array_equal(np_array, array2), "Numpy array could not be loaded")

    def test_save_and_load_pickle(self):
        data1 = np.random.random_sample((3, 128, 128))

        self.experimentLogger.save_pickle(data1, "pickle.pkl")
        np_path = os.path.join(self.experimentLogger.save_dir, "pickle.pkl")
        self.assertTrue(os.path.exists(np_path),
                        "Pickle file could not be saved")

        data2 = self.experimentLogger.load_pickle("pickle.pkl")
        self.assertTrue(np.array_equal(data1, data2), "Pickle file could not be loaded")


if __name__ == '__main__':
    unittest.main()
