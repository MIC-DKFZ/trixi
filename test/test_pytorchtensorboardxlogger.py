import os
import tempfile
import unittest
import shutil
import time

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import misc

from trixi.logger.experiment.experimentlogger import ExperimentLogger
from trixi.logger.tensorboard.pytorchtensorboardxlogger import PytorchTensorboardXLogger
from trixi.util.config import Config


class TestPytorchTensorboardXLogger(unittest.TestCase):

    def setUp(self):
        self.test_dir = tempfile.gettempdir()
        self.logger = PytorchTensorboardXLogger(self.test_dir)

    def tearDown(self):
        self.logger.close()
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_show_text(self):
        self.logger.show_text("Test")

    def test_show_image(self):
        image = np.random.random_sample((3, 128, 128))
        self.logger.show_image(image)

    def test_show_images(self):
        images = np.random.random_sample((16, 3, 128, 128))
        self.logger.show_images(images)

    def test_show_image_grid(self):
        images = np.random.random_sample((16, 3, 128, 128))
        self.logger.show_image_grid(images)

    def test_show_barplot(self):
        tensor = np.random.random_sample(5)
        self.logger.show_barplot(tensor)

    def test_show_lineplot(self):
        x = [0, 1, 2, 3, 4, 5]
        y = np.random.random_sample(6)
        self.logger.show_lineplot(y, x, name="lineplot")

    def test_show_piechart(self):
        array = np.random.random_sample(5)
        self.logger.show_piechart(array, name="piechart")

    def test_show_scatterplot(self):
        array = np.random.random_sample((5, 2))
        self.logger.show_scatterplot(array, name="scatterplot")

    def test_show_value(self):
        val = np.random.random_sample(1)
        self.logger.show_value(val, "value")
        val = np.random.random_sample(1)
        self.logger.show_value(val, "value")
        val = np.random.random_sample(1)
        self.logger.show_value(val, "value", counter=4)
        
        val = np.random.random_sample(1)
        self.logger.show_value(val, "value1", tag="xD")
        val = np.random.random_sample(1)
        self.logger.show_value(val, "value2", tag="xD")
        val = np.random.random_sample(1)
        self.logger.show_value(val, "value1", counter=4, tag="xD")

    def test_show_pr_curve(self):
        self.logger.show_pr_curve(np.random.rand(100), np.random.randint(2, size=100))

    def test_show_embedding(self):
        import torch
        label_img = torch.rand(100, 3, 10, 32)
        for i in range(100):
            label_img[i]*=i/100.0
        self.logger.show_embedding(torch.randn(100, 5), label_img=label_img)

    def test_show_histogram(self):
        self.logger.show_histogram(np.random.rand(100))
        


if __name__ == '__main__':
    unittest.main()
