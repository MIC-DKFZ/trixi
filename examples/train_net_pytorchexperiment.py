import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision.transforms import transforms

from trixi.experiment import PytorchExperiment
from trixi.util import Config


### Get Params

###############################################
#
# Basic Training script for a neural network. Does not really train, and the images look fancy, but only for
# trixi demo (including error plots, image plots, model checkpoint storing).
#
################################################


class Net(nn.Module):
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
        return x


c = Config(
    name="MNIST-Example",
    base_dir="MNIST_experiment/",
    n_epochs=5,
    batch_size=64,
    log_interval=100,
    lr=1e-5,
    use_cuda=True,

)


class MNISTExperiment(PytorchExperiment):

    def setup(self):

        self.elog.print(self.config)

        self.device = torch.device("cuda" if self.config.use_cuda else "cpu")

        ### Get Dataset
        transf = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        self.dataset_train = torchvision.datasets.MNIST(root=self.config.base_dir + "data/", download=True,
                                                        transform=transf, train=True)
        self.dataset_test = torchvision.datasets.MNIST(root=self.config.base_dir + "data/", download=True,
                                                       transform=transf, train=False)

        data_loader_kwargs = {'num_workers': 1, 'pin_memory': True} if self.config.use_cuda else {}
        self.train_loader = torch.utils.data.DataLoader(self.dataset_train, batch_size=self.config.batch_size,
                                                        shuffle=True, **data_loader_kwargs)
        self.test_loader = torch.utils.data.DataLoader(self.dataset_test, batch_size=self.config.batch_size,
                                                       shuffle=True, **data_loader_kwargs)

        ### Models
        self.model = Net()
        self.model.to(self.device)

        ### Optimizers
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config.lr)

        if "load_path" in self.config:
            self.load_checkpoint(path=self.config.load_path, name="")

        ### Criterion
        self.criterion = torch.nn.CrossEntropyLoss()

    def train(self, epoch):
        self.model.train()
        for batch_idx, (data, target) in enumerate(self.train_loader):
            data, target = data.to(self.device), target.to(self.device)

            self.optimizer.zero_grad()
            output = self.model(data)

            loss = self.criterion(output, target)
            loss.backward()
            self.optimizer.step()

            if batch_idx % self.config.log_interval == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(self.train_loader.dataset),
                           100. * batch_idx / len(self.train_loader), loss.item()))

    def validate(self, epoch):
        self.model.eval()
        val_loss = 0
        correct = 0
        with torch.no_grad():
            for data, target in self.test_loader:
                data, target = data.to(self.device), target.to(self.device)

                output = self.model(data)
                val_loss += self.criterion(output, target).item()  # sum up batch loss

                pred = output.max(1, keepdim=True)[1]  # get the index of the max log-probability
                correct += pred.eq(target.view_as(pred)).sum().item()

        val_loss /= len(self.test_loader.dataset)
        print('\nValidation set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            val_loss, correct, len(self.test_loader.dataset),
            100. * correct / len(self.test_loader.dataset)))


if __name__ == '__main__':
    mnist_exp = MNISTExperiment(config=c, globs=globals())
    mnist_exp.run()

    load_path = os.path.join(mnist_exp.elog.checkpoint_dir, "checkpoint_last.pth.tar")
    c.load_path = load_path

    mnist_exp_continued = MNISTExperiment(config=c, globs=globals())
    mnist_exp_continued.run()
