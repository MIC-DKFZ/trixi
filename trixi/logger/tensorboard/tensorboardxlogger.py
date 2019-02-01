import atexit
import os
from collections import defaultdict

from tensorboardX import SummaryWriter

from trixi.logger.plt.numpyseabornplotlogger import NumpySeabornPlotLogger
from trixi.logger.abstractlogger import convert_params
from trixi.util.util import np_make_grid


class TensorboardXLogger(NumpySeabornPlotLogger):
    """Abstract interface for visual logger."""

    def __init__(self, target_dir, *args, **kwargs):

        super(TensorboardXLogger, self).__init__(*args, **kwargs)

        os.makedirs(target_dir, exist_ok=True)

        self.writer = SummaryWriter(target_dir)
        self.val_dict = defaultdict(int)

        atexit.register(self.writer.close)

    def show_image(self, image, name="Image", counter=None, **kwargs):
        """Sends a image to tensorboard."""

        if counter is not None:
            self.val_dict[f"{name}-image"] = counter
        else:
            self.val_dict[f"{name}-image"] += 1

        self.writer.add_image(name, image, global_step=self.val_dict[f"{name}-image"])

    def show_images(self, images, name="Images", counter=None, **kwargs):
        """Sends a list of images to tensorboard."""

        if counter is not None:
            self.val_dict[f"{name}-image"] = counter
        else:
            self.val_dict[f"{name}-image"] += 1

        self.writer.add_image(name, images, global_step=self.val_dict[f"{name}-image"])

    @convert_params
    def show_value(self, value, name="Value", counter=None, tag=None, **kwargs):
        """Sends a value as scalar to tensorboard."""

        if tag is None:
            key = name + "-" + name
        else:
            key = tag + "-" + name

        if counter is not None:
            self.val_dict[f"{key}-image"] = counter
        else:
            self.val_dict[f"{key}-image"] += 1

        if tag is not None:
            self.writer.add_scalars(tag, {name: value}, global_step=self.val_dict[f"{key}-image"])
            self.writer.scalar_dict = {}
        else:
            self.writer.add_scalar(name, value, global_step=self.val_dict[f"{key}-image"])

    def show_text(self, text, name="Text", counter=None, **kwargs):
        """Sends a text to tensorboard."""

        if counter is not None:
            self.val_dict[f"{name}-text"] = counter
        else:
            self.val_dict[f"{name}-text"] += 1

        self.writer.add_text(name, text, global_step=self.val_dict[f"{name}-text"])

    @convert_params
    def show_image_grid(self, image_array, name="Image-Grid", counter=None, image_args=None, *args, **kwargs):
        """
        Sends an array of images to tensorboard.

        Args:
            image_array (np.narray / torch.tensor): Image array/ tensor which will be sent as an image grid
        """

        if image_args is None:
            image_args = dict(nrow=8, padding=2, normalize=False, scale_each=False, pad_value=0)

        if counter is not None:
            self.val_dict[f"{name}-image"] = counter
        else:
            self.val_dict[f"{name}-image"] += 1

        grid = np_make_grid(image_array, **image_args)
        self.writer.add_image(tag=name, img_tensor=grid, global_step=self.val_dict[f"{name}-image"])
        self.val_dict[name] += 1

    @convert_params
    def show_barplot(self, array, name="barplot", counter=None, *args, **kwargs):
        """
        Sends a barplot to tensorboard.

        Args:
            array: array of shape NxM where N is the number of rows and M is the number of elements in the row.
            name: The name of the figure
            counter: Global step value to record

        """

        if counter is not None:
            self.val_dict[f"{name}-figure"] = counter
        else:
            self.val_dict[f"{name}-figure"] += 1

        figure = super().show_barplot(array, name, *args, **kwargs)
        self.writer.add_figure(tag=name, figure=figure, global_step=self.val_dict[f"{name}-figure"])

    @convert_params
    def show_lineplot(self, y_vals, x_vals=None, name="lineplot", counter=None, *args, **kwargs):
        """
        Sends a lineplot to tensorboard.

        Args:
            y_vals: Array of shape MxN , where M is the number of points and N is the number of different line
            x_vals: Has to have the same shape as Y: MxN. For each point in Y it gives the corresponding X value (if
            not set the points are assumed to be equally distributed in the interval [0, 1] )
            name: The name of the figure
            counter: Global step value to record

        """

        if counter is not None:
            self.val_dict[f"{name}-figure"] = counter
        else:
            self.val_dict[f"{name}-figure"] += 1

        figure = super().show_lineplot(y_vals, x_vals, name, *args, **kwargs)
        self.writer.add_figure(tag=name, figure=figure, global_step=self.val_dict[f"{name}-figure"])

    @convert_params
    def show_scatterplot(self, array, name="scatterplot", counter=None, *args, **kwargs):
        """
        Sends a scatterplot to tensorboard.

        Args:
            array: A 2d array with size N x dim, where each element i \in N at X[i] results in a a 2d (if dim = 2)/
            3d (if dim = 3) point.
            name: The name of the figure
            counter: Global step value to record

        """

        if counter is not None:
            self.val_dict[f"{name}-figure"] = counter
        else:
            self.val_dict[f"{name}-figure"] += 1

        figure = super().show_scatterplot(array, name, *args, **kwargs)
        self.writer.add_figure(tag=name, figure=figure, global_step=self.val_dict[f"{name}-figure"])

    @convert_params
    def show_piechart(self, array, name="piechart", counter=None, *args, **kwargs):
        """
        Sends a piechart tensorboard.

        Args:
            array: Array of positive integers. Each integer will be presented as a part of the pie (with the total
            as the sum of all integers)
            name: The name of the figure
            counter: Global step value to record

        """

        if counter is not None:
            self.val_dict[f"{name}-figure"] = counter
        else:
            self.val_dict[f"{name}-figure"] += 1

        figure = super().show_piechart(array, name, *args, **kwargs)
        self.writer.add_figure(tag=name, figure=figure, global_step=self.val_dict[f"{name}-figure"])

    def show_embedding(self, tensor, labels=None, name='default', label_img=None, counter=None,
                       *args, **kwargs):
        """
        Displays a tensor a an embedding

        Args:
            tensor: Tensor to be embedded an then displayed
            labels: List of labels, each element will be convert to string
            name: The name for the embedding
            label_img: Images to be displayed at the embedding points
            counter:  Global step value to record

        """

        if counter is not None:
            self.val_dict[f"{name}-embedding"] = counter
        else:
            self.val_dict[f"{name}-embedding"] += 1

        self.writer.add_embedding(mat=tensor, metadata=labels, label_img=label_img, tag=name, global_step=self.val_dict[f"{name}-embedding"])

    def show_histogram(self, array, name="Histogram", counter=None, *args, **kwargs):
        """
        Plots a histogram in the tensorboard histrogram plugin

        Args:
            counter:  Global step value to record
            name: Data identifier
            array  (torch.Tensor, numpy.array) – Values to build histogram

        """

        if counter is not None:
            self.val_dict[f"{name}-histogram"] = counter
        else:
            self.val_dict[f"{name}-histogram"] += 1

        self.writer.add_histogram(tag=name, values=array, global_step=self.val_dict[f"{name}-histogram"])

    def show_pr_curve(self, tensor, labels, name="pr-curve", counter=None, *args, **kwargs):
        """
        Displays a precision recall curve given a tensor with scores and the coresponding labels

        Args:
            tensor: Tensor with scores (e.g class probability )
            labels: Labels of the samples to which the scores match
            name: The name of the plot
        """

        if counter is not None:
            self.val_dict[f"{name}-pr-curve"] = counter
        else:
            self.val_dict[f"{name}-pr-curve"] += 1

        self.writer.add_pr_curve(tag=name, labels=labels, predictions=tensor, global_step=self.val_dict[f"{name}-pr-curve"])

    def close(self):
        self.writer.close()
