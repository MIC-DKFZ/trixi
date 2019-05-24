import atexit
import os
from collections import defaultdict

from torch.utils.tensorboard import SummaryWriter

from trixi.logger.plt.numpyseabornplotlogger import NumpySeabornPlotLogger
from trixi.logger.abstractlogger import convert_params
from trixi.util.util import np_make_grid


class TensorboardXLogger(NumpySeabornPlotLogger):
    """Logger that uses tensorboardX to log to Tensorboard."""

    def __init__(self, target_dir, *args, **kwargs):

        super(TensorboardXLogger, self).__init__(*args, **kwargs)

        os.makedirs(target_dir, exist_ok=True)

        self.writer = SummaryWriter(target_dir)
        self.val_dict = defaultdict(int)

        atexit.register(self.writer.close)

    def show_image(self, image, name="Image", counter=None, **kwargs):
        """
        Sends an image to tensorboard.

        Args:
            image (np.narray/torch.tensor): Image array/tensor which will be sent
            name (str): Identifier for the image
            counter (int): Global step value
        """

        if counter is not None:
            self.val_dict["{}-image".format(name)] = counter
        else:
            self.val_dict["{}-image".format(name)] += 1

        self.writer.add_image(name, image, global_step=self.val_dict["{}-image".format(name)])

    def show_images(self, images, name="Images", counter=None, **kwargs):
        """
        Sends multiple images to tensorboard.

        Args:
            image (np.narray/torch.tensor): Image array/tensor which will be sent (NxCxHxW)
            name (str): Identifier for the images
            counter (int): Global step value
        """

        if counter is not None:
            self.val_dict["{}-image".format(name)] = counter
        else:
            self.val_dict["{}-image".format(name)] += 1

        self.writer.add_images(name, images, global_step=self.val_dict["{}-image".format(name)])

    @convert_params
    def show_value(self, value, name="Value", counter=None, tag=None, **kwargs):
        """
        Sends a scalar value to tensorboard.

        Args:
            value (numeric): Value to be sent
            name (str): Identifier for the value
            counter (int): Global step value
            tag (str): Identifier for the frame (values with the same tag will be shown in the same graph)
        """

        if tag is None:
            key = name + "-" + name
        else:
            key = tag + "-" + name

        if counter is not None:
            self.val_dict["{}-image".format(key)] = counter
        else:
            self.val_dict["{}-image".format(key)] += 1

        if tag is not None:
            self.writer.add_scalars(tag, {name: value}, global_step=self.val_dict["{}-image".format(key)])
            self.writer.scalar_dict = {}
        else:
            self.writer.add_scalar(name, value, global_step=self.val_dict["{}-image".format(key)])

    def show_text(self, text, name="Text", counter=None, **kwargs):
        """
        Sends text to tensorboard.

        Args:
            text (str): Text to be sent
            name (str): Identifier for the text
            counter (int): Global step value
        """

        if counter is not None:
            self.val_dict["{}-text".format(name)] = counter
        else:
            self.val_dict["{}-text".format(name)] += 1

        self.writer.add_text(name, text, global_step=self.val_dict["{}-text".format(name)])

    @convert_params
    def show_image_grid(self, image_array, name="Image-Grid", counter=None, nrow=8, padding=2,
                        normalize=False, range=None, scale_each=False, pad_value=0,
                        *args, **kwargs):
        """
        Sends an array of images to tensorboard as a grid. Like :meth:`.show_image`, but generates
        image grid before.

        Args:
            image_array (np.narray/torch.tensor): Image array/tensor which will be sent as an image grid
            name (str): Identifier for the image grid
            counter (int): Global step value
            nrow (int): Items per row in grid
            padding (int): Padding between images in grid
            normalize (bool): Normalize images in grid
            range (tuple): Tuple (min, max), so images will be normalized to this range
            scale_each (bool): If True, each image will be normalized separately instead of using
                min and max of whole tensor
            pad_value (float): Fill padding with this value
        """

        image_args = dict(nrow=nrow,
                          padding=padding,
                          normalize=normalize,
                          range=range,
                          scale_each=scale_each,
                          pad_value=pad_value)

        if counter is not None:
            self.val_dict["{}-image".format(name)] = counter
        else:
            self.val_dict["{}-image".format(name)] += 1

        grid = np_make_grid(image_array, **image_args)
        self.writer.add_image(tag=name, img_tensor=grid, global_step=self.val_dict["{}-image".format(name)])
        self.val_dict[name] += 1

    @convert_params
    def show_barplot(self, array, name="barplot", counter=None, *args, **kwargs):
        """
        Sends a barplot to tensorboard.

        Args:
            array (np.array/torch.tensor): array of shape NxM where N is the number of rows and M is the number of elements in the row.
            name (str): The name of the figure
            counter (int): Global step value to record

        """

        if counter is not None:
            self.val_dict["{}-figure".format(name)] = counter
        else:
            self.val_dict["{}-figure".format(name)] += 1

        figure = super().show_barplot(array, name, *args, **kwargs)
        self.writer.add_figure(tag=name, figure=figure, global_step=self.val_dict["{}-figure".format(name)])

    @convert_params
    def show_lineplot(self, y_vals, x_vals=None, name="lineplot", counter=None, *args, **kwargs):
        """
        Sends a lineplot to tensorboard.

        Args:
            y_vals (np.array/torch.tensor): Array of shape MxN , where M is the number of points and N is the number of different line
            x_vals (np.array/torch.tensor): Has to have the same shape as Y: MxN. For each point in Y it gives the corresponding X value (if
                not set the points are assumed to be equally distributed in the interval [0, 1])
            name (str): The name of the figure
            counter (int): Global step value to record

        """

        if counter is not None:
            self.val_dict["{}-figure".format(name)] = counter
        else:
            self.val_dict["{}-figure".format(name)] += 1

        figure = super().show_lineplot(y_vals, x_vals, name, *args, **kwargs)
        self.writer.add_figure(tag=name, figure=figure, global_step=self.val_dict["{}-figure".format(name)])

    @convert_params
    def show_scatterplot(self, array, name="scatterplot", counter=None, *args, **kwargs):
        """
        Sends a scatterplot to tensorboard.

        Args:
            array (np.array/torch.tensor): An array with size N x dim, where each element i \in N` at X[i] results in a 2D
                (if dim = 2) or 3D (if dim = 3) point.
            name (str): The name of the figure
            counter (int): Global step value to record

        """

        if counter is not None:
            self.val_dict["{}-figure".format(name)] = counter
        else:
            self.val_dict["{}-figure".format(name)] += 1

        figure = super().show_scatterplot(array, name, *args, **kwargs)
        self.writer.add_figure(tag=name, figure=figure, global_step=self.val_dict["{}-figure".format(name)])

    @convert_params
    def show_piechart(self, array, name="piechart", counter=None, *args, **kwargs):
        """
        Sends a piechart tensorboard.

        Args:
            array (np.array/torch.tensor): Array of positive integers. Each integer will be
                presented as a part of the pie (with the total as the sum of all integers)
            name (str): The name of the figure
            counter (int): Global step value to record

        """

        if counter is not None:
            self.val_dict["{}-figure".format(name)] = counter
        else:
            self.val_dict["{}-figure".format(name)] += 1

        figure = super().show_piechart(array, name, *args, **kwargs)
        self.writer.add_figure(tag=name, figure=figure, global_step=self.val_dict["{}-figure".format(name)])

    def show_embedding(self, tensor, labels=None, name='default', label_img=None, counter=None,
                       *args, **kwargs):
        """
        Displays an embedding of a tensor (for more details see tensorboardX)

        Args:
            tensor (torch.tensor/np.array): Tensor to be embedded and then displayed
            labels (list): List of labels, each element will be converted to string
            name (str): The name for the embedding
            label_img (torch.tensor): Images to be displayed at the embedding points
            counter (int):  Global step value to record

        """

        if counter is not None:
            self.val_dict["{}-embedding".format(name)] = counter
        else:
            self.val_dict["{}-embedding".format(name)] += 1

        self.writer.add_embedding(mat=tensor, metadata=labels, label_img=label_img, tag=name, global_step=self.val_dict["{}-embedding".format(name)])

    def show_histogram(self, array, name="Histogram", counter=None, *args, **kwargs):
        """
        Plots a histogram in the tensorboard histrogram plugin

        Args:
            array (torch.tensor/np.array): Values to build histogram
            name (str): Data identifier
            counter (int):  Global step value to record

        """

        if counter is not None:
            self.val_dict["{}-histogram".format(name)] = counter
        else:
            self.val_dict["{}-histogram".format(name)] += 1

        self.writer.add_histogram(tag=name, values=array, global_step=self.val_dict["{}-histogram".format(name)])

    def show_pr_curve(self, tensor, labels, name="pr-curve", counter=None, *args, **kwargs):
        """
        Displays a precision recall curve given a tensor with scores and the corresponding labels

        Args:
            tensor (torch.tensor/np.array): Tensor with scores (e.g class probabilities)
            labels (list): Labels of the samples to which the scores match
            name (str): The name of the plot
            counter (int): Global step value
        """

        if counter is not None:
            self.val_dict["{}-pr-curve".format(name)] = counter
        else:
            self.val_dict["{}-pr-curve".format(name)] += 1

        self.writer.add_pr_curve(tag=name, labels=labels, predictions=tensor, global_step=self.val_dict["{}-pr-curve".format(name)])

    def close(self):
        self.writer.close()
