
import os
import unittest
import shutil
import time

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import trixi


test_dir = "test_dir"


class TestExperimentLogger(unittest.TestCase):

    def setUp(self):
        remove_if_exists(test_dir)
        self.experimentLogger = trixi.PytorchExperimentLogger(experiment_name="test",
                                                                  base_dir=test_dir,
                                                                  folder_format="{experiment_name}")

    def test_created_base_test_folder(self):
        self.assertTrue(os.path.isdir("test_dir"), "test directory not created")

    def test_two_experiment_loggers_same_test_dir_no_run_number_throws_error(self):
        with self.assertRaises(FileExistsError):
            trixi.PytorchExperimentLogger(experiment_name="test",
                                              base_dir=test_dir,
                                              folder_format="{experiment_name}")

    def test_net_save_and_load(self):
        net = Net()
        save_fn = self.experimentLogger.get_save_checkpoint_fn(net=net)
        save_fn(n_iter=1)
        # give some time before loading, so saving has finished
        time.sleep(1)
        net2 = Net()
        diff_conv1 = np.abs(np.sum(net.conv1.bias.data.numpy() - net2.conv1.bias.data.numpy()))
        self.assertTrue(diff_conv1 > 1e-3, "conv1 bias values have not been initialized differently")
        self.experimentLogger.load_last_checkpoint(dir="test_dir/test/checkpoint/", net=net2)
        np.testing.assert_allclose(net.conv1.bias.data.numpy(), net2.conv1.bias.data.numpy(),
                                   err_msg="loading checkpoint did not restore model values")

    def test_net_save_and_load_with_optimizer(self):
        self._test_load_save(test_with_cuda=False)

    def test_net_save_and_load_with_optimizer_with_cuda(self):
        self._test_load_save(test_with_cuda=True)

    def _test_load_save(self, test_with_cuda):
        # small testing net
        net = Net()
        # fake values for fake step
        random_input = np_to_var(np.random.randn(28*28).reshape((1,1, 28, 28)))
        fake_labels = np_to_var(np.array([2])).long()

        if test_with_cuda:
            net.cuda()
            random_input, fake_labels = random_input.cuda(), fake_labels.cuda()

        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(net.parameters(), lr=1e-9, eps=1e-7, weight_decay=1e-4, betas=(0.3,0.4))

        # do a fake forward/backward/step pass
        pred = net(random_input)
        err = criterion(pred, fake_labels)
        err.backward()
        optimizer.step()

        # save and load network and optimizer
        save_fn = self.experimentLogger.get_save_checkpoint_fn(net=net, optimizer=optimizer)
        save_fn(n_iter=1)
        # give some time before loading, so saving has finished
        time.sleep(1)

        loaded_network = Net()
        if test_with_cuda:
            loaded_network.cuda()

        loaded_optimizer = torch.optim.Adam(loaded_network.parameters(), lr=1e-10)
        self.experimentLogger.load_last_checkpoint(dir="test_dir/test/checkpoint/", net=loaded_network, optimizer=loaded_optimizer)

        saved_pg = optimizer.param_groups[0]
        loaded_pg = loaded_optimizer.param_groups[0]

        # assert optimizer values have been correctly saved/loaded
        self.assertAlmostEqual(saved_pg["lr"], loaded_pg["lr"], "learning rate not correctly saved")
        self.assertAlmostEqual(saved_pg["eps"], loaded_pg["eps"], "eps not correctly saved")
        self.assertAlmostEqual(saved_pg["weight_decay"], loaded_pg["weight_decay"], "weight_decay not correctly saved")
        self.assertAlmostEqual(saved_pg["betas"][0], loaded_pg["betas"][0], "beta 0 not correctly saved")
        self.assertAlmostEqual(saved_pg["betas"][1], loaded_pg["betas"][1], "beta 1 not correctly saved")
        for saved_parameter, loaded_parameter in zip(saved_pg["params"], loaded_pg["params"]):
            np.testing.assert_allclose(saved_parameter.data.cpu().numpy(), loaded_parameter.data.cpu().numpy(),
                                       err_msg="loading checkpoint did not restore optimizer param values")

        # do a fake forward/backward/step pass with loaded stuff
        pred = loaded_network(random_input)
        err = criterion(pred, fake_labels)
        err.backward()
        loaded_optimizer.step()  # this could raise an error, which is why there is no assert but just the execution

    def tearDown(self):
        remove_if_exists(test_dir)


def remove_if_exists(folder_name):
    if os.path.exists(folder_name):
        shutil.rmtree(folder_name)


def np_to_var(array):
    tensor = torch.from_numpy(array).float()
    var = torch.autograd.Variable(tensor, volatile=False)
    return var


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
