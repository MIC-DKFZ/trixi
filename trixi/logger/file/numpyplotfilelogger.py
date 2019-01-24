import os

from trixi.logger.abstractlogger import convert_params
from trixi.logger.plt.numpyseabornplotlogger import NumpySeabornPlotLogger
from trixi.util import savefig_and_close


# this is just to turn threaded into non-threaded
def threaded(func):
    return func


class NumpyPlotFileLogger(NumpySeabornPlotLogger):
    """
    NumpyPlotFileLogger is a logger, which can plot/ interpret numpy array as different types (images, lineplots, ...)
    into an image and plot directory. For the plotting it builds up on the NumpySeabornPlotLogger.

    """

    def __init__(self, img_dir, plot_dir, **kwargs):
        """
        Initializes a numpy plot file logger to plot images and plots into an image and plot directory

        Args:
            img_dir: The directory to store images in
            plot_dir: The directory to store plots in
        """
        super(NumpyPlotFileLogger, self).__init__(**kwargs)
        self.img_dir = img_dir
        self.plot_dir = plot_dir

    @convert_params
    def show_image(self, image, name, file_format=".png", *args, **kwargs):
        """
        Method which stores an image as a image file

        Args:
            image: Numpy array-image
            name: file-name
            file_format: output-image file format

        """
        figure = NumpySeabornPlotLogger.show_image(self, image, name, show=False)
        outname = os.path.join(self.img_dir, name) + file_format
        os.makedirs(os.path.dirname(outname), exist_ok=True)
        threaded(savefig_and_close)(figure, outname)

    @convert_params
    def show_value(self, value, name, counter=None, tag=None, file_format=".png", *args, **kwargs):
        """
        Method which logs a value as a line plot

        Args:
            value: Value (y-axis value) you want to display/ plot/ store
            name: Name of the value (will also be the filename if no tag is given)
            counter: counter, which tells the number of the sample (with the same name --> filename) (x-axis value)
            tag: Tag, grouping similar values. Values with the same tag will be plotted in the same plot
            file_format: output-image file format

        Returns:

        """
        figure = NumpySeabornPlotLogger.show_value(self, value, name, counter, tag, show=False)
        if tag is None:
            outname = os.path.join(self.plot_dir, name) + file_format
        else:
            outname = os.path.join(self.plot_dir, tag) + file_format
        os.makedirs(os.path.dirname(outname), exist_ok=True)
        threaded(savefig_and_close)(figure, outname)

    @convert_params
    def show_barplot(self, array, name, file_format=".png", *args, **kwargs):
        """
        Method which creates and stores a barplot

        Args:
            array: Array of values you want to plot
            name: file-name
            file_format: output-image (plot) file format
        """
        figure = NumpySeabornPlotLogger.show_barplot(self, array, name, show=False)
        outname = os.path.join(self.plot_dir, name) + file_format
        os.makedirs(os.path.dirname(outname), exist_ok=True)
        threaded(savefig_and_close)(figure, outname)

    @convert_params
    def show_boxplot(self, array, name, file_format=".png", *args, **kwargs):
        """
        Method which creates and stores a boxplot

        Args:
            array: Array of values you want to plot
            name: file-name
            file_format: output-image (plot) file format
        """
        figure = NumpySeabornPlotLogger.show_boxplot(self, array, name, show=False, **kwargs)
        outname = os.path.join(self.plot_dir, name) + file_format
        os.makedirs(os.path.dirname(outname), exist_ok=True)
        threaded(savefig_and_close)(figure, outname)

    @convert_params
    def show_lineplot(self, y_vals, x_vals, name, file_format=".png", *args, **kwargs):
        """
        Method which creates and stores a lineplot

        Args:
            y_vals: Array of y values
            x_vals: Array of corresponding x-values
            name: file-name
            file_format: output-image (plot) file format
        """
        figure = NumpySeabornPlotLogger.show_lineplot(self, y_vals, x_vals, name, show=False)
        outname = os.path.join(self.plot_dir, name) + file_format
        os.makedirs(os.path.dirname(outname), exist_ok=True)
        threaded(savefig_and_close)(figure, outname)

    @convert_params
    def show_scatterplot(self, array, name, file_format=".png", *args, **kwargs):
        """
        Method which creates and stores a scatter

        Args:
            array: Array of values you want to plot
            name: file-name
            file_format: output-image (plot) file format
        """
        figure = NumpySeabornPlotLogger.show_scatterplot(self, array, name, show=False)
        outname = os.path.join(self.plot_dir, name) + file_format
        os.makedirs(os.path.dirname(outname), exist_ok=True)
        threaded(savefig_and_close)(figure, outname)

    @convert_params
    def show_piechart(self, array, name, file_format=".png", *args, **kwargs):
        """
        Method which creates and stores a piechart

        Args:
            array: Array of values you want to plot
            name: file-name
            file_format: output-image (plot) file format
        """
        figure = NumpySeabornPlotLogger.show_piechart(self, array, name, show=False)
        outname = os.path.join(self.plot_dir, name) + file_format
        os.makedirs(os.path.dirname(outname), exist_ok=True)
        threaded(savefig_and_close)(figure, outname)
