from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import seaborn

from trixi.logger.abstractlogger import AbstractLogger, convert_params
from trixi.util.util import chw_to_hwc


class NumpySeabornPlotLogger(AbstractLogger):
    """
    Visual logger, inherits the AbstractLogger and plots/ logs numpy arrays/ values as matplotlib / seaborn plots.
    """

    def __init__(self, **kwargs):
        """
        Initializes a new NumpySeabornPlotLogger
        """
        super(NumpySeabornPlotLogger, self).__init__(**kwargs)

        self.values = defaultdict(lambda: defaultdict(list))
        self.max_values = defaultdict(int)

    @convert_params
    def show_image(self, image, name=None, show=True, *args, **kwargs):
        """
        Create an image figure

        Args:
            image: The image array to be displayed
            name: The name of the image window
            show: Flag if it should also display the figure (result might also depend on the matplotlib backend )

        Returns:
            A matplotlib figure

        """
        figure = self.get_figure(name)
        plt.clf()

        image = chw_to_hwc(image)

        plt.imshow(image)
        plt.axis("off")
        if show:
            plt.show(block=False)
            plt.pause(0.01)

        return figure

    @convert_params
    def show_value(self, value, name, counter=None, tag=None, show=True, *args, **kwargs):
        """
       Creates a line plot that is automatically appended with new values and returns it as a figure.

       Args:
           value: Value to be plotted / appended to the graph (y-axis value)
           name: The name of the window
           counter: counter, which tells the number of the sample (with the same name) (x-axis value)
           tag: Tag, grouping similar values. Values with the same tag will be plotted in the same plot
           show: Flag if it should also display the figure (result might also depend on the matplotlib backend )

        Returns:
            A matplotlib figure

               """
        figure = self.get_figure(tag)
        plt.clf()

        seaborn.set_style("whitegrid")

        if tag is None:
            tag = name

        if counter is None:
            counter = len(self.values[tag][name]) + 1

        max_val = max(self.max_values[tag], counter)
        self.max_values[tag] = max_val
        self.values[tag][name].append((value, max_val))

        for y_tag in self.values[tag]:

            y, x = zip(*self.values[tag][y_tag])
            plt.plot(x, y, label=y_tag)

        if show:
            plt.show(block=False)
            plt.pause(0.01)

        plt.legend()
        plt.ylabel(name)

        return figure

    @convert_params
    def show_barplot(self, array, name=None, show=True, *args, **kwargs):
        """
        Creates a bar plot figure from an array

        Args:
            array: array of shape NxM where N is the number of rows and M is the number of elements in the row.
            name: The name of the figure
            show: Flag if it should also display the figure (result might also depend on the matplotlib backend )

        Returns:
            A matplotlib figure
        """

        figure = self.get_figure(name)

        y = array
        x = list(range(1, len(y) + 1))

        seaborn.set_style("whitegrid")

        ax = seaborn.barplot(y=y, x=x)

        if show:
            plt.show(block=False)
            plt.pause(0.01)

        return figure

    @convert_params
    def show_boxplot(self, array, name, show=True, *args, **kwargs):
        """
        Creates a box plot figure from an array

        Args:
            array: array of shape NxM where N is the number of rows and M is the number of elements in the row.
            name: The name of the figure
            show: Flag if it should also display the figure (result might also depend on the matplotlib backend )

        Returns:
            A matplotlib figure
        """

        figure = self.get_figure(name)
        seaborn.set_style("whitegrid")

        ax = seaborn.boxplot(data=array)

        handles, _ = ax.get_legend_handles_labels()
        try:
            legend = kwargs['opts']['legend']
            ax.legend(handles, legend)
        except KeyError: # if no legend is defined
            pass
        if show:
            plt.show(block=False)
            plt.pause(0.01)

        return figure

    @convert_params
    def show_lineplot(self, y_vals, x_vals=None, name=None, show=True, *args, **kwargs):
        """
        Creates a line plot figure with (multiple) lines plot, given values Y (and optional the corresponding X values)

        Args:
            y_vals: Array of shape MxN , where M is the number of points and N is the number of different line
            x_vals: Has to have the same shape as Y: MxN. For each point in Y it gives the corresponding X value (if
            not set the points are assumed to be equally distributed in the interval [0, 1] )
            name: The name of the figure
            show: Flag if it should also display the figure (result might also depend on the matplotlib backend )

        Returns:
            A matplotlib figure
        """

        figure = self.get_figure(name)
        plt.clf()

        seaborn.set_style("whitegrid")

        if x_vals is None:
            x_vals = list(range(len(y_vals)))

        plt.plot(x_vals, y_vals)

        if show:
            plt.show(block=False)
            plt.pause(0.01)

        return figure

    @convert_params
    def show_scatterplot(self, array, name=None, show=True, *args, **kwargs):
        """
        Creates a scatter plot figure with the points given in array

        Args:
            array: A 2d array with size N x dim, where each element i \in N at X[i] results in a a 2d (if dim = 2)/
            3d (if dim = 3) point.
            name: The name of the figure
            show: Flag if it should also display the figure (result might also depend on the matplotlib backend )

        Returns:
            A matplotlib figure
        """

        if not isinstance(array, np.ndarray):
            raise TypeError("Array must be numpy arrays (this class is called NUMPY seaborn logger, and seaborn"
                            " can only handle numpy arrays -.- .__. )")
        if len(array.shape) != 2:
            raise ValueError("Array must be 2D for scatterplot")
        if array.shape[1] != 2:
            raise ValueError("Array must be 2D and have x,y pairs in the 2nd dim for scatterplot")

        x, y = zip(*array)
        x, y = np.asarray(x), np.asarray(y)

        figure = self.get_figure(name)
        plt.clf()

        ax = seaborn.regplot(x=x, y=y, fit_reg=False)

        if show:
            plt.show(block=False)
            plt.pause(0.01)

        return figure

    @convert_params
    def show_piechart(self, array, name=None, show=True, *args, **kwargs):
        """
        Creates a scatter plot figure

        Args:
            array: Array of positive integers. Each integer will be presented as a part of the pie (with the total
            as the sum of all integers)
            name: The name of the figure
            show: Flag if it should also display the figure (result might also depend on the matplotlib backend )

        Returns:
            A matplotlib figure
        """

        figure = self.get_figure(name)
        plt.clf()

        plt.pie(array)

        if show:
            plt.show(block=False)
            plt.pause(0.01)

        return figure

    def get_figure(self, name):
        """
        Returns a figure with a given name as identifier.

        If no figure yet exists with the name a new one is created. Otherwise the existing one is returned

        Args:
            name: Name of the figure

        Returns:
            A figure with the given name
        """

        return plt.figure(name)
