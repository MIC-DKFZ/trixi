import os

from trixi.logger.plt.numpyseabornplotlogger import NumpySeabornPlotLogger
from trixi.logger.abstractlogger import convert_params
from trixi.util import savefig_and_close


# this is just to turn threaded into non-threaded
def threaded(func):
    return func


class NumpyPlotFileLogger(NumpySeabornPlotLogger):

    def __init__(self, img_dir, plot_dir, **kwargs):
        super(NumpyPlotFileLogger, self).__init__(**kwargs)
        self.img_dir = img_dir
        self.plot_dir = plot_dir

    @convert_params
    def show_image(self, image, name, file_format=".png", *args, **kwargs):
        """Abstract method which should handle and somehow log/ store an image"""
        figure = NumpySeabornPlotLogger.show_image(self, image, name, show=False)
        outname = os.path.join(self.image_dir, name) + file_format
        os.makedirs(os.path.dirname(outname), exist_ok=True)
        threaded(savefig_and_close)(figure, outname)

    @convert_params
    def show_value(self, value, name, counter=None, tag=None, file_format=".png", *args, **kwargs):
        """Abstract method which should handle and somehow log/ store a value"""
        figure = NumpySeabornPlotLogger.show_value(self, value, name, counter, tag, show=False)
        outname = os.path.join(self.plot_dir, name) + file_format
        os.makedirs(os.path.dirname(outname), exist_ok=True)
        threaded(savefig_and_close)(figure, outname)

    @convert_params
    def show_barplot(self, array, name, file_format=".png", *args, **kwargs):
        """Abstract method which should handle and somehow log/ store a barplot"""
        figure = NumpySeabornPlotLogger.show_barplot(self, array, name, show=False)
        outname = os.path.join(self.plot_dir, name) + file_format
        os.makedirs(os.path.dirname(outname), exist_ok=True)
        threaded(savefig_and_close)(figure, outname)

    @convert_params
    def show_lineplot(self, y_vals, x_vals, name, file_format=".png", *args, **kwargs):
        """Abstract method which should handle and somehow log/ store a lineplot"""
        figure = NumpySeabornPlotLogger.show_lineplot(self, x_vals, y_vals, name, show=False)
        outname = os.path.join(self.plot_dir, name) + file_format
        os.makedirs(os.path.dirname(outname), exist_ok=True)
        threaded(savefig_and_close)(figure, outname)

    @convert_params
    def show_scatterplot(self, array, name, file_format=".png", *args, **kwargs):
        """Abstract method which should handle and somehow log/ store a scatterplot"""
        figure = NumpySeabornPlotLogger.show_scatterplot(self, array, name, show=False)
        outname = os.path.join(self.plot_dir, name) + file_format
        os.makedirs(os.path.dirname(outname), exist_ok=True)
        threaded(savefig_and_close)(figure, outname)

    @convert_params
    def show_piechart(self, array, name, file_format=".png", *args, **kwargs):
        """Abstract method which should handle and somehow log/ store a piechart"""
        figure = NumpySeabornPlotLogger.show_piechart(self, array, name, show=False)
        outname = os.path.join(self.plot_dir, name) + file_format
        os.makedirs(os.path.dirname(outname), exist_ok=True)
        threaded(savefig_and_close)(figure, outname)
