import os
import tempfile
import unittest
import shutil
import time

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from trixi.logger.message.slackmessagelogger import SlackMessageLogger
from trixi.logger.plt.numpyseabornimageplotlogger import NumpySeabornImagePlotLogger
from trixi.logger.visdom import PytorchVisdomLogger
from trixi.logger.visdom.numpyvisdomlogger import start_visdom

#
# class TestSlackMessageLogger(unittest.TestCase):
#
#     @classmethod
#     def setUpClass(cls):
#         super(TestSlackMessageLogger, cls).setUpClass()
#
#
#     def setUp(self):
#         self.logger = SlackMessageLogger(token="", user_email="")
#
#     def test_show_image(self):
#         image = np.random.random_sample((3, 128, 128))
#         figure_image = self.logger.show_image(image, "image")
#
#     def test_show_barplot(self):
#         tensor = np.random.random_sample(5)
#         figure_image = self.logger.show_barplot(tensor, name="barplot")
#
#     def test_show_lineplot(self):
#         x = [0, 1, 2, 3, 4, 5]
#         y = np.random.random_sample(6)
#         figure_image = self.logger.show_lineplot(y, x, name="lineplot1")
#
#     def test_show_piechart(self):
#         array = np.random.random_sample(5)
#         figure_image = self.logger.show_piechart(array, name="piechart")
#
#     def test_show_scatterplot(self):
#         array = np.random.random_sample((5, 2))
#         figure_image = self.logger.show_scatterplot(array, name="scatterplot")
#
#     def test_show_value(self):
#         val = np.random.random_sample(1)
#         figure_image = self.logger.show_value(val, "value")
#
#         val = np.random.random_sample(1)
#         figure_image = self.logger.show_value(val, "value")
#
#         val = np.random.random_sample(1)
#         figure_image = self.logger.show_value(val, "value", counter=4)
#
# if __name__ == '__main__':
#     unittest.main()
