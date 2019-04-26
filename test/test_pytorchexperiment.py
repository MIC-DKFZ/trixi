import json
import os
import tempfile
import unittest
import shutil
import time

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from trixi.experiment import PytorchExperiment
from trixi.util import ResultLogDict


class TestPytorchExperiment(unittest.TestCase):

    def setUp(self):
        self.test_dir = tempfile.gettempdir()
        self.experiment = PytorchExperiment(name="test_experiment", base_dir=self.test_dir, n_epochs=10)

    def tearDown(self):
        self.experiment._exp_state = "Ended"
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_train(self):
        self.cntr = []
        self.experiment.train = lambda epoch: self.cntr.append(0)
        self.experiment.run()
        self.assertTrue(len(self.cntr) == 10, "Did not call train for the right number of epochs")

    def test_validate(self):
        self.cntr = []
        self.experiment.validate = lambda epoch: self.cntr.append(0)
        self.experiment.run()
        self.assertTrue(len(self.cntr) == 10, "Did not call train for the right number of epochs")

    def test_update_attributes(self):
        results2 = ResultLogDict("results-log.json", base_dir=self.test_dir)
        results2["test"] = 0
        self.experiment.test_var = "test"

        self.experiment.update_attributes({"results": results2, "test_var": "test2"})

        self.assertTrue(self.experiment.test_var == "test2", "Could not update simple var in update_attributes")
        self.assertTrue(self.experiment.results['test'] == 0, "Could not result dict var in update_attributes")

    def test_get_pytorch_modules(self):
        module = Net()
        self.experiment.net = module

        pyt_module = self.experiment.get_pytorch_modules()

        self.assertTrue("net" in pyt_module, "Name for module could not be correctly returned in get_pytorch_modules")
        self.assertTrue(pyt_module['net'] == module, "Module could not be correctly returned in  get_pytorch_modules")

    def test_get_pytorch_optimizers(self):
        optimizer = torch.optim.Adam(params=[torch.tensor(1)])
        self.experiment.optim = optimizer

        pyt_optim = self.experiment.get_pytorch_optimizers()

        self.assertTrue("optim" in pyt_optim, "Name for optimizer could not be correctly returned in "
                                              "get_pytorch_optimizers")
        self.assertTrue(pyt_optim['optim'] == optimizer, "Optimizer could not be correctly returned in  "
                                                         "get_pytorch_optimizers")

    def test_get_simple_vars(self):
        self.experiment.a = "test"
        self.experiment.b = 1
        self.experiment.c = True
        self.experiment.d = (1, 2, 3, 4)

        var = self.experiment.get_simple_variables()

        self.assertTrue("a" in var and "b" in var and "c" in var and "d" in var,
                        "Names for attributes could not be correctly returned in get_simple_variables")
        self.assertTrue(var['a'] == "test" and var['b'] == 1 and var['c'] == True and var['d'] == (1, 2, 3, 4),
                        "variables could not be correctly returned in get_simple_variables")

    def test_get_pytorch_vars(self):
        var = torch.tensor(1)
        self.experiment.pyt_var = var

        pyt_vars = self.experiment.get_pytorch_variables()

        self.assertTrue("pyt_var" in pyt_vars, "Name for variable could not be correctly returned in "
                                               "get_pytorch_variables")
        self.assertTrue(pyt_vars['pyt_var'] == var, "Variable could not be correctly returned in  "
                                                    "get_pytorch_variables")

    def test_save_results(self):
        self.experiment.add_result(0, "test")
        self.experiment.save_results("results-test.json")

        self.assertTrue(os.path.exists(os.path.join(self.experiment.elog.result_dir, "results-test.json")),
                        "result file could not be stored")

        with open(os.path.join(self.experiment.elog.result_dir, "results-test.json"), "r") as f:
            content = f.read()
            self.assertTrue("test" in content and "0" in content, "results content not sucessfully saved")

    def test_save_and_load_checkpoints(self):
        net = Net()
        optim = torch.optim.Adam(params=net.parameters())
        ptvar = torch.tensor(1)
        svar = "test"
        lvar = (1, 2, 3, 4)

        self.experiment.net = net
        self.experiment.optim = optim
        self.experiment.ptvar = ptvar
        self.experiment.svar = svar
        self.experiment.lvar = lvar
        self.experiment.results["test"] = 1

        self.experiment.save_checkpoint(name="test_checkpoint")

        self.experiment.net = Net()
        self.experiment.optim = torch.optim.Adam(params=Net().parameters(), lr=1)
        self.experiment.ptvar = torch.tensor(0)
        self.experiment.svar = "test2"
        self.experiment.lvar = (-1, -2)
        self.experiment.results["test"] = 0

        self.experiment.load_checkpoint(name="test_checkpoint")

        self.assertTrue((list(self.experiment.net.parameters())[0] - list(net.parameters())[0]).sum().item() < 0.00001,
                        "Net could not be restored from checkpoint")
        self.assertTrue(
            self.experiment.optim.state_dict()['param_groups'][0]['lr'] == optim.state_dict()['param_groups'][0]['lr'],
            "optim could not be restored from checkpoint")
        self.assertTrue(self.experiment.ptvar == ptvar.item(), "ptvar could not be restored from checkpoint")
        self.assertTrue(self.experiment.svar == svar, "svar could not be restored from checkpoint")
        self.assertTrue(self.experiment.lvar == lvar, "lvar could not be restored from checkpoint")
        self.assertTrue(self.experiment.results["test"] == 1, "Results could not be restored from checkpoint")

    def test_add_results(self):
        self.experiment.add_result(name="test", value=1)

        self.assertTrue(self.experiment.results["test"] == 1,
                        "Result was not added")
        self.assertTrue(self.experiment.get_result("test") == 1,
                        "Result was not added")

        self.assertTrue(os.path.exists(os.path.join(self.experiment.elog.result_dir, "results-log.json")),
                        "result file could not be stored")

        with open(os.path.join(self.experiment.elog.result_dir, "results-log.json"), "r") as f:
            content = f.read()
        self.assertTrue("test" in content and "1" in content, "results content not sucessfully saved")

    def test_save_tmp_results(self):
        self.experiment.add_result(name="test", value=1)
        self.experiment.run()

        self.assertTrue(os.path.exists(os.path.join(self.experiment.elog.result_dir, "results.json")),
                        "result file could not be stored")

        with open(os.path.join(self.experiment.elog.result_dir, "results.json"), "r") as f:
            content = f.read()
        self.assertTrue("test" in content and "1" in content, "results content not sucessfully temporarily saved")

    def test_save_tmp_checkpoint(self):
        self.experiment.test_var = "test"
        self.experiment.run()

        time.sleep(5)

        self.assertTrue(os.path.exists(os.path.join(self.experiment.elog.checkpoint_dir, "checkpoint_current.pth.tar")),
                        "Temp Checkpoint file could not be stored")
        self.assertTrue(os.path.exists(os.path.join(self.experiment.elog.checkpoint_dir, "checkpoint_last.pth.tar")),
                        "Last Checkpoint file could not be stored")

        self.experiment.test_var = "test2"
        self.experiment.load_checkpoint("checkpoint_current")
        self.assertTrue(self.experiment.test_var == "test",
                        "Temp Checkpoint file loading could not sucessful")
        self.experiment.test_var = "test2"
        self.experiment.load_checkpoint("checkpoint_last")
        self.assertTrue(self.experiment.test_var == "test",
                        "Temp Checkpoint file loading could not sucessful")

    def test_save_exp_info(self):
        self.experiment.run()

        with open(os.path.join(self.experiment.elog.config_dir, "exp.json"), "r") as f:
            exp_info = json.load(f)

        self.assertTrue(exp_info['epoch'] == 10, "Epoch not sucessfully stored in exp info")
        self.assertTrue(exp_info['name'] == 'test_experiment', "Name not sucessfully stored in exp info")
        self.assertTrue(exp_info['state'] == 'Trained', "State not sucessfully stored as 'Trained' in exp info")

    def test_print(self):
        self.experiment.print("0o zD 0o")
        with open(os.path.join(self.experiment.elog.log_dir, "default.log"), "r") as f:
            content = f.read()
        self.assertTrue("0o zD 0o" in content, "Print not sucessfully saved")

    def test_resume(self):
        # TODO

        self.cntr = []
        self.experiment.train = lambda epoch: self.cntr.append(0)
        self.experiment.end = lambda: time.sleep(2)
        self.experiment.run()
        self.assertTrue(len(self.cntr) == 10, "Did not call train for the right number of epochs")

        exp2 = PytorchExperiment(name="test-exp2", base_dir=self.test_dir, resume=self.experiment.elog.work_dir)
        exp3 = PytorchExperiment(name="test-exp2", base_dir=self.test_dir, resume=self.experiment.elog.work_dir,
                                 resume_reset_epochs=False)

        exp2.prepare_resume()
        exp3.prepare_resume()

        self.assertTrue(exp2.exp_name == 'test_experiment', "Did not restore exp_name")
        self.assertTrue(exp2._epoch_idx == 0, "Did not reset epochs")
        self.assertTrue(exp3._epoch_idx == 10, "Did reset epochs")

        exp2.train = lambda epoch: self.cntr.append(0)
        exp3.train = lambda epoch: self.cntr.append(0)
        exp2.run()
        self.assertTrue(len(self.cntr) == 20, "Did call not train for exp2")
        exp3.run()
        self.assertTrue(len(self.cntr) == 20, "Did call train for exp3")

        exp2._exp_state = "Ended"
        exp3._exp_state = "Ended"



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
