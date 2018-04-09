import os

use_agg = True

import matplotlib
if use_agg: matplotlib.use("Agg")

from vislogger.numpyseabornplotlogger import NumpySeabornPlotLogger
from vislogger.abstractlogger import convert_params, threaded
from vislogger.util import savefig_and_close


class NumpyPlotFileLogger(NumpySeabornPlotLogger):

    def __init__(self, img_dir, plot_dir, **kwargs):
        super(NumpyPlotFileLogger, self).__init__(**kwargs)
        self.img_dir = img_dir
        self.plot_dir = plot_dir

    @convert_params
    def show_image(self, image, name, file_format=".png", *args, **kwargs):
        """Abstract method which should handle and somehow log/ store an image"""
        figure = NumpySeabornPlotLogger.show_image(self, image, name, show=False)
        threaded(savefig_and_close)(figure, os.path.join(self.image_dir, name) + file_format)

    @convert_params
    def show_value(self, value, name, count=None, tag=None, file_format=".png", *args, **kwargs):
        """Abstract method which should handle and somehow log/ store a value"""
        figure = NumpySeabornPlotLogger.show_value(self, value, name, count, tag, show=False)
        threaded(savefig_and_close)(figure, os.path.join(self.plot_dir, name) + file_format)

    @convert_params
    def show_barplot(self, array, name, file_format=".png", *args, **kwargs):
        """Abstract method which should handle and somehow log/ store a barplot"""
        figure = NumpySeabornPlotLogger.show_barplot(self, array, name, show=False)
        threaded(savefig_and_close)(figure, os.path.join(self.plot_dir, name) + file_format)

    @convert_params
    def show_lineplot(self, y_vals, x_vals, name, file_format=".png", *args, **kwargs):
        """Abstract method which should handle and somehow log/ store a lineplot"""
        figure = NumpySeabornPlotLogger.show_lineplot(self, x_vals, y_vals, name, show=False)
        threaded(savefig_and_close)(figure, os.path.join(self.plot_dir, name) + file_format)

    @convert_params
    def show_scatterplot(self, array, name, file_format=".png", *args, **kwargs):
        """Abstract method which should handle and somehow log/ store a scatterplot"""
        figure = NumpySeabornPlotLogger.show_scatterplot(self, array, name, show=False)
        threaded(savefig_and_close)(figure, os.path.join(self.plot_dir, name) + file_format)

    @convert_params
    def show_piechart(self, array, name, file_format=".png", *args, **kwargs):
        """Abstract method which should handle and somehow log/ store a piechart"""
        figure = NumpySeabornPlotLogger.show_piechart(self, array, name, show=False)
        threaded(savefig_and_close)(figure, os.path.join(self.plot_dir, name) + file_format)
