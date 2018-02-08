import os

use_agg = True

import matplotlib
if use_agg:
    matplotlib.use("Agg")

from vislogger.filelogger import FileLogger
from vislogger.numpyseabornlogger import NumpySeabornLogger
from vislogger.abstractvisuallogger import convert_params


class NumpyFileLogger(NumpySeabornLogger, FileLogger):
    def __init__(self, path, **kwargs):
        super(NumpyFileLogger, self).__init__(path=path, **kwargs)

    @convert_params
    def show_image(self, image, name, file_format=".png", *args, **kwargs):
        """Abstract method which should handle and somehow log/ store an image"""
        figure = NumpySeabornLogger.show_image(self, image, name, show=False)
        figure.savefig(os.path.join(self.image_dir, name) + file_format)

    @convert_params
    def show_value(self, value, name, file_format=".png", *args, **kwargs):
        """Abstract method which should handle and somehow log/ store a value"""
        figure = NumpySeabornLogger.show_value(self, value, name, show=False)
        figure.savefig(os.path.join(self.plot_dir, name) + file_format)

    @convert_params
    def show_text(self, text, *args, **kwargs):
        """Abstract method which should handle and somehow log/ store a text"""
        FileLogger.show_text(text)

    @convert_params
    def show_barplot(self, array, name, file_format=".png", *args, **kwargs):
        """Abstract method which should handle and somehow log/ store a barplot"""
        figure = NumpySeabornLogger.show_barplot(self, array, name, show=False)
        figure.savefig(os.path.join(self.plot_dir, name) + file_format)

    @convert_params
    def show_lineplot(self, y_vals, x_vals, name, file_format=".png", *args, **kwargs):
        """Abstract method which should handle and somehow log/ store a lineplot"""
        figure = NumpySeabornLogger.show_lineplot(self, x_vals, y_vals, name, show=False)
        figure.savefig(os.path.join(self.plot_dir, name) + file_format)

    @convert_params
    def show_scatterplot(self, array, name, file_format=".png", *args, **kwargs):
        """Abstract method which should handle and somehow log/ store a scatterplot"""
        figure = NumpySeabornLogger.show_scatterplot(self, array, name, show=False)
        figure.savefig(os.path.join(self.plot_dir, name) + file_format)

    @convert_params
    def show_piechart(self, array, name, file_format=".png", *args, **kwargs):
        """Abstract method which should handle and somehow log/ store a piechart"""
        figure = NumpySeabornLogger.show_piechart(self, array, name, show=False)
        figure.savefig(os.path.join(self.plot_dir, name) + file_format)
