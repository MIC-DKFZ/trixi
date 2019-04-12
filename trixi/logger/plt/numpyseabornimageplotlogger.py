from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import seaborn
from trixi.logger.plt.numpyseabornplotlogger import NumpySeabornPlotLogger

from trixi.logger.abstractlogger import AbstractLogger, convert_params
from trixi.util.util import chw_to_hwc, figure_to_image


class NumpySeabornImagePlotLogger(NumpySeabornPlotLogger):
    """
    Wrapper around :class:`.NumpySeabornPlotLogger` that renders figures into numpy arrays.
    """

    def __init__(self, **kwargs):
        """
        Initializes a new NumpySeabornPlotLogger
        """
        super(NumpySeabornImagePlotLogger, self).__init__(**kwargs)

    @convert_params
    def show_image(self, image, name=None, *args, **kwargs):
        """
        Create an image figure

        Args:
            image: The image array to be displayed
            name: The name of the image window

        Returns:
            A numpy array image of the figure

        """
        # figure = super().show_image(image, name, show=False, *args, **kwargs)
        # figure_image = figure_to_image(figure)

        return image

    @convert_params
    def show_value(self, value, name, counter=None, tag=None, *args, **kwargs):
        """
       Creates a line plot that is automatically appended with new values and returns it as a figure.

       Args:
           value: Value to be plotted / appended to the graph (y-axis value)
           name: The name of the window
           counter: counter, which tells the number of the sample (with the same name) (x-axis value)
           tag: Tag, grouping similar values. Values with the same tag will be plotted in the same plot

        Returns:
            A numpy array image of the figure

        """
        figure = super().show_value(value, name, show=False, counter=counter, tag=tag, *args, **kwargs)
        figure_image = figure_to_image(figure)

        return figure_image

    @convert_params
    def show_barplot(self, array, name=None, *args, **kwargs):
        """
        Creates a bar plot figure from an array

        Args:
            array: array of shape NxM where N is the number of rows and M is the number of elements in the row.
            name: The name of the figure

        Returns:
            A numpy array image of the figure
        """

        figure = super().show_barplot(array, name, show=False, *args, **kwargs)
        figure_image = figure_to_image(figure)

        return figure_image

    @convert_params
    def show_lineplot(self, y_vals, x_vals=None, name=None, *args, **kwargs):
        """
        Creates a line plot figure with (multiple) lines plot, given values Y (and optional the corresponding X values)

        Args:
            y_vals: Array of shape MxN , where M is the number of points and N is the number of different line
            x_vals: Has to have the same shape as Y: MxN. For each point in Y it gives the corresponding X value (if
            not set the points are assumed to be equally distributed in the interval [0, 1] )
            name: The name of the figure

        Returns:
            A numpy array image of the figure
        """

        figure = super().show_lineplot(y_vals=y_vals, x_vals=x_vals, name=name, show=False, *args, **kwargs)
        figure_image = figure_to_image(figure)

        return figure_image

    @convert_params
    def show_scatterplot(self, array, name=None, *args, **kwargs):
        """
        Creates a scatter plot figure with the points given in array

        Args:
            array: A 2d array with size N x dim, where each element i \in N at X[i] results in a a 2d (if dim = 2)/
            3d (if dim = 3) point.
            name: The name of the figure

         Returns:
            A numpy array image of the figure
        """

        figure = super().show_scatterplot(array, name, show=False, *args, **kwargs)
        figure_image = figure_to_image(figure)

        return figure_image

    @convert_params
    def show_piechart(self, array, name=None, *args, **kwargs):
        """
        Creates a scatter plot figure

        Args:
            array: Array of positive integers. Each integer will be presented as a part of the pie (with the total
            as the sum of all integers)
            name: The name of the figure

        Returns:
            A numpy array image of the figure
        """

        figure = super().show_piechart(array, name, show=False, *args, **kwargs)
        figure_image = figure_to_image(figure)

        return figure_image
