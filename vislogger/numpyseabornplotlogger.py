from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import seaborn

from vislogger.abstractlogger import AbstractLogger
from vislogger.abstractlogger import convert_params


class NumpySeabornPlotLogger(AbstractLogger):
    """
    Visual logger, inherits the AbstractLogger and plots/ logs numpy arrays/ values as matplotlib / seaborn plots.
    """

    def __init__(self, **kwargs):
        super(NumpySeabornPlotLogger, self).__init__(**kwargs)

        self.values = defaultdict(lambda: defaultdict(list))
        self.max_values = defaultdict(int)

    @convert_params
    def show_image(self, image, name, show=True, *args, **kwargs):
        """A method which creats an image plot"""

        figure = self.get_figure(name)
        plt.clf()

        plt.imshow(image)
        plt.axis("off")
        if show:
            plt.show(block=False)
            plt.pause(0.01)

        return figure

    @convert_params
    def show_value(self, value, name, counter=None, tag=None, show=True, *args, **kwargs):
        """A method which should handle and somehow log/ store a value"""

        figure = self.get_figure(name)
        plt.clf()

        seaborn.set_style("whitegrid")

        if tag is None:
            tag = name

        if counter is None:
            counter = len(self.values[name][tag]) + 1

        max_val = max(self.max_values[name], counter)
        self.max_values[name] = max_val
        self.values[name][tag].append((value, max_val))

        for y_name in self.values[name]:

            y, x = zip(*self.values[name][y_name])
            plt.plot(x, y)

        if show:
            plt.show(block=False)
            plt.pause(0.01)

        return figure

    @convert_params
    def show_barplot(self, array, name, show=True, *args, **kwargs):
        """A method which should handle and somehow log/ store a barplot"""

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
    def show_lineplot(self, y_vals, x_vals, name, show=True, *args, **kwargs):
        """A method which should handle and somehow log/ store a lineplot"""

        figure = self.get_figure(name)
        plt.clf()

        seaborn.set_style("whitegrid")

        plt.plot(x_vals, y_vals)

        if show:
            plt.show(block=False)
            plt.pause(0.01)

        return figure

    @convert_params
    def show_scatterplot(self, array, name, show=True, *args, **kwargs):
        """A method which should handle and somehow log/ store a scatterplot"""

        if not isinstance(array, np.ndarray):
            raise TypeError("Array must be numpy arrays (this class is called NUMPY seaborn logger, and seaborn"
                            " can only handle numpy arrays -.- .__. )")

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
    def show_piechart(self, array, name, show=True, *args, **kwargs):
        """A method which should handle and somehow log/ store a piechart"""

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

        :param name: Name of the figure
        :return: A figure with the given name
        """

        return plt.figure(name)
