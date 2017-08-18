from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import seaborn

from vislogger.abstractvisuallogger import AbstractVisualLogger
from vislogger.abstractvisuallogger import convert_params


class NumpySeabornLogger(AbstractVisualLogger):
    """
    Visual logger, inherits the AbstractVisualLogger and plots/ logs numpy arrays/ values as matplotlib / seaborn plots.
    """

    def __init__(self, **kwargs):
        super(NumpySeabornLogger, self).__init__(**kwargs)

        self.figures = {}
        self.values = defaultdict(lambda: defaultdict(list))
        self.max_values = defaultdict(int)

    @convert_params
    def show_image(self, image, name, show=True, *args, **kwargs):
        """A method which creats an image plot"""
        figure = self.get_figure(name)
        plt.imshow(image)
        plt.axis("off")
        if show:
            plt.show(block=False)
            plt.pause(0.01)

        return figure

    @convert_params
    def show_value(self, value, name, line_name=None, show=True, *args, **kwargs):
        """A method which should handle and somehow log/ store a value"""
        figure = self.get_figure(name)
        plt.clf()

        seaborn.set_style("whitegrid")


        if line_name is None:
            line_name = name

        max_val = max(self.max_values[name], len(self.values[name][line_name]) + 1)
        self.max_values[name] = max_val
        self.values[name][line_name].append((value, max_val))

        for y_name in self.values[name]:

            y, x = zip(*self.values[name][y_name])

            plt.plot(x, y)

        if show:
            plt.show(block=False)
            plt.pause(0.01)

        return figure



    @convert_params
    def show_barplot(self, data, name, show=True, *args, **kwargs):
        """A method which should handle and somehow log/ store a barplot"""
        figure = self.get_figure(name)

        y = data
        x = list(range(1, len(y) + 1))

        seaborn.set_style("whitegrid")

        ax = seaborn.barplot(y=y, x=x)

        if show:
            plt.show(block=False)
            plt.pause(0.01)

        return figure

    @convert_params
    def show_lineplot(self, x, y, name, show=True, *args, **kwargs):
        """A method which should handle and somehow log/ store a lineplot"""

        figure = self.get_figure(name)
        plt.clf()

        seaborn.set_style("whitegrid")

        plt.plot(x, y)

        if show:
            plt.show(block=False)
            plt.pause(0.01)

        return figure

    @convert_params
    def show_scatterplot(self, name, x, y, show=True, *args, **kwargs):
        """A method which should handle and somehow log/ store a scatterplot"""

        if not isinstance(x, np.ndarray) or not isinstance(y, np.ndarray):
            raise TypeError("x and y must be numpy arrays (this class is called NUMPY seaborn logger, and seaborn"
                            " can only handle numpy array -.- .__. )")

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
        #        if name not in self.figures:
        self.figures[name] = plt.figure(name)

        return self.figures[name]
