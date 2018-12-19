import os
import tempfile
import unittest
import shutil
import time

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from trixi.logger.visdom import PytorchVisdomLogger
from trixi.logger.visdom.numpyvisdomlogger import start_visdom


class TestPytorchVisdomLogger(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        super(TestPytorchVisdomLogger, cls).setUpClass()
        try:
            start_visdom()
        except:
            print("Could not start visdom, it might be already running.")

    def setUp(self):
        self.visdomLogger = PytorchVisdomLogger()

    def test_show_image(self):
        image = np.random.random_sample((3, 128, 128))
        tensor = torch.from_numpy(image)
        self.visdomLogger._NumpyVisdomLogger__show_image(tensor.numpy(), "image")

    def test_show_images(self):
        images = np.random.random_sample((4, 3, 128, 128))
        tensors = torch.from_numpy(images)
        self.visdomLogger._NumpyVisdomLogger__show_images(tensors.numpy(), "image")

    def test_show_image_grid(self):
        images = np.random.random_sample((4, 3, 128, 128))
        tensor = torch.from_numpy(images)
        self.visdomLogger._PytorchVisdomLogger__show_image_grid(tensor, "image_grid")

    def test_show_image_grid_heatmap(self):
        images = np.random.random_sample((4, 3, 128, 128))
        self.visdomLogger._PytorchVisdomLogger__show_image_grid_heatmap(images, name="image_grid_heatmap")

    def test_show_barplot(self):
        tensor = torch.from_numpy(np.random.random_sample(5))
        self.visdomLogger.show_barplot(tensor, name="barplot")
        self.visdomLogger._NumpyVisdomLogger__show_barplot(tensor.numpy(), name="barplot")

    def test_show_lineplot(self):
        x = [0, 1, 2, 3, 4, 5]
        y = np.random.random_sample(6)
        self.visdomLogger.show_lineplot(y, x, name="lineplot1")
        self.visdomLogger._NumpyVisdomLogger__show_lineplot(y, x, name="lineplot1")

    def test_show_piechart(self):
        array = torch.from_numpy(np.random.random_sample(5))
        self.visdomLogger.show_piechart(array, name="piechart")
        self.visdomLogger._NumpyVisdomLogger__show_piechart(array, name="piechart")

    def test_show_scatterplot(self):
        array = torch.from_numpy(np.random.random_sample((5, 2)))
        self.visdomLogger.show_scatterplot(array, name="scatterplot")
        self.visdomLogger._NumpyVisdomLogger__show_scatterplot(array.numpy(), name="scatterplot")

    def test_show_value(self):
        val = torch.from_numpy(np.random.random_sample(1))
        self.visdomLogger.show_value(val, "value")
        self.visdomLogger._NumpyVisdomLogger__show_value(val.numpy(), "value")

        val = torch.from_numpy(np.random.random_sample(1))
        self.visdomLogger.show_value(val, "value")

        val = torch.from_numpy(np.random.random_sample(1))
        self.visdomLogger.show_value(val, "value", counter=4)

    def test_show_text(self):
        text = "\nTest 4 fun: zD ;-D 0o"
        self.visdomLogger.show_text(text)
        self.visdomLogger._NumpyVisdomLogger__show_text(text)

    def test_get_roc_curve(self):
        array = np.random.random_sample(100)
        labels = np.random.choice((0, 1), 100)

        self.visdomLogger.show_roc_curve(array, labels, name="roc")

    def test_get_pr_curve(self):
        array = np.random.random_sample(100)
        labels = np.random.choice((0, 1), 100)

        self.visdomLogger.show_roc_curve(array, labels, name="pr")

    def test_get_classification_metric(self):
        array = np.random.random_sample(100)
        labels = np.random.choice((0, 1), 100)

        self.visdomLogger.show_classification_metrics(array, labels, metric=("roc-auc", "pr-score"),
                                                      name="classification-metrics")

    def test_show_image_gradient(self):
        net = Net()
        random_input = torch.from_numpy(np.random.randn(28 * 28).reshape((1, 1, 28, 28))).float()
        fake_labels = torch.from_numpy(np.array([2])).long()
        criterion = torch.nn.CrossEntropyLoss()

        err_fn = lambda x: criterion(x, fake_labels)

        self.visdomLogger.show_image_gradient(name="grads-vanilla", model=net, inpt=random_input, err_fn=err_fn,
                                              grad_type="vanilla")
        time.sleep(1)

        self.visdomLogger.show_image_gradient(name="grads-svanilla", model=net, inpt=random_input, err_fn=err_fn,
                                              grad_type="smooth-vanilla")
        time.sleep(1)

        self.visdomLogger.show_image_gradient(name="grads-guided", model=net, inpt=random_input, err_fn=err_fn,
                                              grad_type="guided")
        time.sleep(1)

        self.visdomLogger.show_image_gradient(name="grads-sguided", model=net, inpt=random_input, err_fn=err_fn,
                                              grad_type="smooth-guided")
        time.sleep(1)

    def test_plot_model_structure(self):
        net = Net()
        self.visdomLogger.plot_model_structure(net, (1, 1, 28, 28))


    def test_plot_model_statistics(self):
        net = Net()
        self.visdomLogger.plot_model_statistics(net, plot_grad=False)
        self.visdomLogger.plot_model_statistics(net, plot_grad=True)

    def test_show_embedding(self):
        array = torch.from_numpy(np.random.random_sample((100, 100)))
        self.visdomLogger.show_embedding(array, method="tsne")
        self.visdomLogger.show_embedding(array, method="umap")


class Net(nn.Module):
    """
    Small network to test save/load functionality
    """

    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


if __name__ == '__main__':
    unittest.main()
