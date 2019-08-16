import os
import tempfile
import unittest
import shutil
import time
import matplotlib
matplotlib.use("Agg")

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from trixi.logger.experiment import PytorchExperimentLogger


class TestPytorchExperimentLogger(unittest.TestCase):

    def setUp(self):
        self.test_dir = tempfile.gettempdir()
        self.experimentLogger = PytorchExperimentLogger(exp_name="test",
                                                        base_dir=self.test_dir,
                                                        folder_format="{experiment_name}")

    def tearDown(self):
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_show_image(self):
        image = np.random.random_sample((3, 128, 128))
        tensor = torch.from_numpy(image)
        self.experimentLogger.show_image(tensor, "image")

        time.sleep(1)
        self.assertTrue(os.path.exists(os.path.join(self.experimentLogger.img_dir, "image.png")),
                        "Show image could not create image")

    def test_show_images(self):
        images = np.random.random_sample((4, 3, 128, 128))
        tensors = torch.from_numpy(images)
        self.experimentLogger.show_images(tensors, "image")

        time.sleep(1)
        self.assertTrue(len(os.listdir(self.experimentLogger.img_dir)) > 3,
                        "Show images could not create multiple images")

    def test_show_image_grid(self):
        images = np.random.random_sample((4, 3, 128, 128))
        tensor = torch.from_numpy(images)
        self.experimentLogger.show_image_grid(tensor, "image_grid")

        time.sleep(1)
        self.assertTrue(os.path.exists(os.path.join(self.experimentLogger.img_dir, "image_grid.png")),
                        "Show image grid could not create image grid from tensor")

    def test_show_image_grid_heatmap(self):
        images = np.random.random_sample((4, 3, 128, 128))
        tensor = torch.from_numpy(images)
        self.experimentLogger.show_image_grid_heatmap(tensor, name="image_grid_heatmap")

        time.sleep(1)
        self.assertTrue(os.path.exists(os.path.join(self.experimentLogger.img_dir, "image_grid_heatmap.png")),
                        "Show image grid could not create image grid from tensor")

    def test_show_barplot(self):
        tensor = torch.from_numpy(np.random.random_sample(5))
        self.experimentLogger.show_barplot(tensor, "barplot")
        self.assertTrue(os.path.exists(os.path.join(self.experimentLogger.plot_dir, "barplot.png")),
                        "Show barplot could not create barplot")

    def test_net_save_and_load_model(self):
        net = Net()
        self.experimentLogger.save_model(name="model", model=net)
        # give some time before loading, so saving has finished
        time.sleep(1)
        net2 = Net()
        diff_conv1 = np.abs(np.sum(net.conv1.bias.detach().numpy() - net2.conv1.bias.detach().numpy()))
        self.assertTrue(diff_conv1 > 1e-3, "conv1 bias values have not been initialized differently")
        self.experimentLogger.load_model(name="model", model=net2)
        time.sleep(2)

        np.testing.assert_allclose(net.conv1.bias.detach().numpy(), net2.conv1.bias.detach().numpy(),
                                   err_msg="loading model did not restore model values")

    def test_net_save_and_load_checkpoint(self):
        net = Net()
        self.experimentLogger.save_checkpoint(name="checkpoint", net=net)
        # give some time before loading, so saving has finished
        time.sleep(1)
        net2 = Net()
        diff_conv1 = np.abs(np.sum(net.conv1.bias.detach().numpy() - net2.conv1.bias.detach().numpy()))
        self.assertTrue(diff_conv1 > 1e-3, "conv1 bias values have not been initialized differently")
        self.experimentLogger.load_last_checkpoint(dir=self.experimentLogger.checkpoint_dir, net=net2)
        np.testing.assert_allclose(net.conv1.bias.detach().numpy(), net2.conv1.bias.detach().numpy(),
                                   err_msg="loading checkpoint did not restore model values")

    def test_net_save_and_load_checkpoint_with_optimizer(self):
        self._test_load_save_checkpoint(test_with_cuda=False)

    def test_net_save_and_load_checkpoint_with_optimizer_with_cuda(self):
        if torch.cuda.is_available():
            self._test_load_save_checkpoint(test_with_cuda=True)

    def _test_load_save_checkpoint(self, test_with_cuda):
        # small testing net
        net = Net()
        # fake values for fake step
        random_input = torch.from_numpy(np.random.randn(28 * 28).reshape((1, 1, 28, 28))).float()
        fake_labels = torch.from_numpy(np.array([2])).long()

        if test_with_cuda:
            net.cuda()
            random_input, fake_labels = random_input.cuda(), fake_labels.cuda()

        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(net.parameters(), lr=1e-1, eps=1e-7, weight_decay=1e-4, betas=(0.3, 0.4))

        # do a fake forward/backward/step pass
        pred = net(random_input)
        err = criterion(pred, fake_labels)
        err.backward()
        optimizer.step()

        # save and load network and optimizer
        self.experimentLogger.save_checkpoint(name="checkpoint", net=net, optimizer=optimizer)
        # give some time before loading, so saving has finished
        time.sleep(1)

        loaded_network = Net()
        if test_with_cuda:
            loaded_network.cuda()

        loaded_optimizer = torch.optim.Adam(loaded_network.parameters(), lr=1e-1)
        self.experimentLogger.load_last_checkpoint(dir=self.experimentLogger.checkpoint_dir, net=loaded_network,
                                                   optimizer=loaded_optimizer)

        saved_pg = optimizer.param_groups[0]
        loaded_pg = loaded_optimizer.param_groups[0]

        # assert optimizer values have been correctly saved/loaded
        self.assertAlmostEqual(saved_pg["lr"], loaded_pg["lr"], "learning rate not correctly saved")
        self.assertAlmostEqual(saved_pg["eps"], loaded_pg["eps"], "eps not correctly saved")
        self.assertAlmostEqual(saved_pg["weight_decay"], loaded_pg["weight_decay"], "weight_decay not correctly saved")
        self.assertAlmostEqual(saved_pg["betas"][0], loaded_pg["betas"][0], "beta 0 not correctly saved")
        self.assertAlmostEqual(saved_pg["betas"][1], loaded_pg["betas"][1], "beta 1 not correctly saved")
        for saved_parameter, loaded_parameter in zip(saved_pg["params"], loaded_pg["params"]):
            np.testing.assert_allclose(saved_parameter.detach().cpu().numpy(), loaded_parameter.detach().cpu().numpy(),
                                       err_msg="loading checkpoint did not restore optimizer param values")

        # do a fake forward/backward/step pass with loaded stuff
        pred = loaded_network(random_input)
        err = criterion(pred, fake_labels)
        err.backward()
        loaded_optimizer.step()  # this could raise an error, which is why there is no assert but just the execution

    def test_print(self):
        text = "\nTest 4 fun: zD ;-D 0o"
        self.experimentLogger.print(text)
        log_text = ""
        with open(os.path.join(self.experimentLogger.log_dir, "default.log"), 'r') as log_file:
            log_text = log_file.read()
        self.assertTrue(text in log_text)

    def test_get_roc_curve(self):

        try:
            import sklearn

            array = np.random.random_sample(100)
            labels = np.random.choice((0, 1), 100)

            tpr, fpr = self.experimentLogger.get_roc_curve(array, labels)
            self.assertTrue(np.all(tpr >= 0) and np.all(tpr <= 1) and np.all(fpr >= 0) and np.all(fpr <= 1),
                            "Got an invalid tpr, fpr")
        except:
            pass

    def test_get_pr_curve(self):

        try:
            import sklearn

            array = np.random.random_sample(100)
            labels = np.random.choice((0, 1), 100)

            precision, recall = self.experimentLogger.get_pr_curve(array, labels)
            self.assertTrue(np.all(precision >= 0) and np.all(precision <= 1)
                            and np.all(recall >= 0) and np.all(recall <= 1),
                            "Got an invalid precision, recall")
        except:
            pass

    def test_get_classification_metric(self):

        try:
            import sklearn

            array = np.random.random_sample(100)
            labels = np.random.choice((0, 1), 100)

            vals, tags = self.experimentLogger.get_classification_metrics(array, labels,
                                                                          metric=("roc-auc", "pr-score"))

            self.assertTrue("roc-auc" in tags and "pr-score" in tags, "Did not get all classification metrics")
            self.assertTrue(vals[0] >= 0 and vals[0] <= 1
                            and vals[1] >= 0 and vals[1] <= 1,
                            "Got an invalid classification metric values")
        except:
            pass

    def test_show_image_gradient(self):

        net = Net()
        random_input = torch.from_numpy(np.random.randn(28 * 28).reshape((1, 1, 28, 28))).float()
        fake_labels = torch.from_numpy(np.array([2])).long()
        criterion = torch.nn.CrossEntropyLoss()

        def err_fn(x):
            x = net(x)
            return criterion(x, fake_labels)

        self.experimentLogger.show_image_gradient("grads-vanilla", model=net, inpt=random_input, err_fn=err_fn,
                                                  grad_type="vanilla")
        time.sleep(1)
        self.assertTrue(os.path.exists(os.path.join(self.experimentLogger.img_dir, "grads-vanilla.png")),
                        "Could not get vanilla gradients")

        self.experimentLogger.show_image_gradient("grads-svanilla", model=net, inpt=random_input, err_fn=err_fn,
                                                  grad_type="smooth-vanilla")
        time.sleep(1)
        self.assertTrue(os.path.exists(os.path.join(self.experimentLogger.img_dir, "grads-svanilla.png")),
                        "Could not get vanilla gradients")

        self.experimentLogger.show_image_gradient("grads-guided", model=net, inpt=random_input, err_fn=err_fn,
                                                  grad_type="guided")
        time.sleep(1)
        self.assertTrue(os.path.exists(os.path.join(self.experimentLogger.img_dir, "grads-guided.png")),
                        "Could not get vanilla gradients")

        self.experimentLogger.show_image_gradient("grads-sguided", model=net, inpt=random_input, err_fn=err_fn,
                                                  grad_type="smooth-guided")
        time.sleep(1)
        self.assertTrue(os.path.exists(os.path.join(self.experimentLogger.img_dir, "grads-sguided.png")),
                        "Could not get vanilla gradients")


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
