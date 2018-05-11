import torch
import torchvision
from torch.autograd import Variable

import trixi

###############################################
#
# Basic Training script for a neural network. Does not really train, and the images look fancy, but only for
# trixi demo (including error plots, image plots, model checkpoint storing).
#
################################################

### Get Params
param = dict(
    name="AlexNet",
    output_folder="AlexNet/",
    n_epoch=100,
    batch_size=32
)

### Init stuff
vizLog = trixi.PytorchVisdomLogger(name=param["name"], port=8080)
expLog = trixi.PytorchExperimentLogger(experiment_name=param["name"], base_dir=param["output_folder"])
combiLog = trixi.CombinedLogger((vizLog, 1), (expLog, 10))

expLog.print(expLog.base_dir)
expLog.text_logger.log_to(param, "config")

### Get Dataset
dataset = torchvision.datasets.MNIST(root="data/", download=True, transform=torchvision.transforms.ToTensor())
dataloader = torch.utils.data.DataLoader(dataset, batch_size=param["batch_size"])

### Models
alexNet = torchvision.models.alexnet(pretrained=False, num_classes=10)
nets = dict(alexNet=alexNet)

if "load_path" in param:
    nets = expLog.load_last_checkpoint(dir=param["load_path"], **nets)

expLog.text_logger.log_to(nets, "nets")

### Criterion
criterion = torch.nn.CrossEntropyLoss()

### Optimizers
optimizer = torch.optim.Adam(alexNet.parameters(), lr=1e-10)

### Store stuff
store_checkpoint_fn = expLog.get_save_checkpoint_fn(optimizer=optimizer, **nets)
expLog.save_at_exit(optimizer=optimizer, **nets)

### Fitting model
for epoch in range(param["n_epoch"]):

    for batch_idx, data_batch in enumerate(dataloader):

        #######################
        #    Update network   #
        #######################

        alexNet.zero_grad()

        images, labels = data_batch

        current_batch_size = images.size(0)
        images = images.repeat(current_batch_size, 3, 28, 28).resize_(current_batch_size, 3, 224, 224)

        inpt = Variable(images)
        labels = Variable(labels)

        pred = alexNet(inpt)
        err = criterion(pred, labels)

        err.backward()
        optimizer.step()

        #######################
        #      Log results    #

        #######################
        combiLog.show_value(value=err.data[0], name='err')

        log_text = '[%d/%d][%d/%d] Loss: %.4f ' % (epoch, param["n_epoch"], batch_idx, len(dataset), err.data[0])

        expLog.print(log_text)

        if batch_idx % 2 == 0:
            vizLog.show_text(log_text, name="log")
            vizLog.show_progress(epoch, param["n_epoch"])

            combiLog.show_image_grid(images, name="xd", title="Samples", n_iter=batch_idx)

    if epoch % 20 == 0:
        store_checkpoint_fn(n_iter=epoch)
