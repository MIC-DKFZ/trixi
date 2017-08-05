from __future__ import division, print_function

import atexit
import multiprocessing as mp
import sys
import traceback

import numpy as np

from abstractvisuallogger import AbstractVisualLogger, convert_params
from extravisdom import ExtraVisdom


class NumpyVisdomLogger(AbstractVisualLogger):
    """
    Visual logger, inherits the AbstractVisualLogger and plots/ logs numpy arrays/ values on a Visdom server.
    """

    def __init__(self, name, server="http://localhost", port=8097, auto_close=False, **kwargs):
        """
        Creates a new NumpyVisdomLogger object.

        :param name: The name of the visdom environment
        :param server: The address of the visdom server
        :param port: The port of the visdom server
        :param auto_close: Close all objects and kill process at the end of the python script
        """

        super(NumpyVisdomLogger, self).__init__(**kwargs)

        self.name = name
        self.server = server
        self.port = port

        self.vis = ExtraVisdom(env=self.name, server=self.server, port=self.port)

        self._value_counter = dict()
        self._3d_histograms = dict()

        self._queue = mp.Queue()
        self._process = mp.Process(target=self.__show, args=(self._queue,))

        if auto_close:
            atexit.register(self.close_all)
            atexit.register(self.exit)

        self._process.start()

    def __show(self, queue):
        """
        Loop for the internal process to process all visualization tasks

        :param queue: queue with all visualization tasks
        """

        while True:
            vis_task = queue.get()

            try:

                if vis_task["type"] == "image":
                    self.__show_image(**vis_task)
                if vis_task["type"] == "images":
                    self.__show_images(**vis_task)
                if vis_task["type"] == "value":
                    self.__show_value(**vis_task)
                if vis_task["type"] == "text":
                    self.__show_text(**vis_task)
                if vis_task["type"] == "progress":
                    self.__show_progress(**vis_task)
                if vis_task["type"] == "histogram":
                    self.__show_histogram(**vis_task)
                if vis_task["type"] == "histogram_3d":
                    self.__show_histogram_3d(**vis_task)
                if vis_task["type"] == "barplot":
                    self.__show_barplot(**vis_task)
                if vis_task["type"] == "lineplot":
                    self.__show_lineplot(**vis_task)
                if vis_task["type"] == "scatterplot":
                    self.__show_scatterplot(**vis_task)
                if vis_task["type"] == "piechart":
                    self.__show_piechart(**vis_task)

            except:

                error = sys.exc_info()[0]
                msg = traceback.format_exc()
                print("Error {}: {}".format(error, msg))

    @convert_params
    def show_image(self, image, name=None, title=None, caption=None, env_appendix=""):
        """
        Displays an image in a window/pane at the visdom server

        :param image: The image array to be displayed
        :param name: The name of the image window
        :param title: The title of the image window
        :param caption: The of the image, displayed in the window
        :param env_appendix: appendix to the environment name, if used the new env is env+env_appendix
        """

        vis_task = {
            "type": "image",
            "image": image,
            "name": name,
            "title": title,
            "caption": caption,
            "env_appendix": env_appendix
        }
        self._queue.put_nowait(vis_task)

    def __show_image(self, image, name=None, title=None, caption=None, env_appendix="", **kwargs):
        """
        Internal show_image method, called by the internal process.
        This function does all the magic.
        """

        win = self.vis.image(
            img=image,
            win=name,
            env=self.name + env_appendix,
            opts=dict(title=title, caption=caption)
        )

        return win

    @convert_params
    def show_images(self, images, name=None, title=None, caption=None, env_appendix=""):
        """
        Displays multiple images in a window/pane at a visdom server

        :param images: The image array to be displayed
        :param name: The name of the window
        :param title: The title of the image window
        :param caption: The of the image, displayed in the window
        :param env_appendix: appendix to the environment name, if used the new env is env+env_appendix
        """

        vis_task = {
            "type": "images",
            "images": images,
            "name": name,
            "title": title,
            "caption": caption,
            "env_appendix": env_appendix
        }
        self._queue.put_nowait(vis_task)

    def __show_images(self, images, name=None, title=None, caption=None, env_appendix="", **kwargs):
        """
       Internal show_images method, called by the internal process.
       This function does all the magic.
        """

        win = self.vis.images(
            tensor=images,
            win=name,
            env=self.name + env_appendix,
            opts=dict(title=title, caption=caption)
        )

        return win

    @convert_params
    def show_value(self, value, name=None, env_appendix=""):
        """
        Creates a line plot that is automatically appended with new values.

        :param value: Value to be plotted / appended to the graph
        :param name: The name of the window
        :param env_appendix: appendix to the environment name, if used the new env is env+env_appendix
        """

        vis_task = {
            "type": "value",
            "value": value,
            "name": name,
            "env_appendix": env_appendix
        }
        self._queue.put_nowait(vis_task)

    def __show_value(self, value, name=None, env_appendix="", **kwargs):
        """
       Internal show_value method, called by the internal process.
       This function does all the magic.
        """

        value = np.asarray([value])

        if name is not None and name in self._value_counter:
            self._value_counter[name] += 1
            up_str = "append"

        else:
            self._value_counter[name] = np.array([0])
            up_str = None

        win = self.vis.line(
            Y=value,
            X=self._value_counter[name],
            win=name,
            update=up_str,
            env=self.name + env_appendix,
            opts=dict(title=name)
        )

        return win

    @convert_params
    def show_text(self, text, name=None, env_appendix=""):
        """
        Displays a text in a visdom window

        :param text: The text to be displayed
        :param name: The name of the window
        :param env_appendix: appendix to the environment name, if used the new env is env+env_appendix
        """

        vis_task = {
            "type": "text",
            "text": text,
            "name": name,
            "env_appendix": env_appendix
        }
        self._queue.put_nowait(vis_task)

    def __show_text(self, text, name=None, env_appendix="", **kwargs):
        """
       Internal show_text method, called by the internal process.
       This function does all the magic.
        """

        win = self.vis.text(
            text=text,
            win=name,
            env=self.name + env_appendix
        )

        return win

    @convert_params
    def show_progress(self, num, total=None, name=None, env_appendix=""):
        """
        Shows the progress as a pie chart.

        :param num: Current progress. Either a relative value (0 <= num <= 1) or a absolute value if total is given (
        but has to be smaller than total)
        :param total: This if the total number of iterations.
        :param name: The name of the window
        :param env_appendix: appendix to the environment name, if used the new env is env+env_appendix
        """

        vis_task = {
            "type": "progress",
            "num": num,
            "total": total,
            "name": name,
            "env_appendix": env_appendix
        }
        self._queue.put_nowait(vis_task)

    def __show_progress(self, num, total=None, name=None, env_appendix="", **kwargs):
        """
       Internal show_progress method, called by the internal process.
       This function does all the magic.
        """

        if total is None:
            if 0 <= num <= 1:
                total = 1000.
                num = 1000. * num
            else:
                raise AttributeError("Either num has to be a ratio between 0 and 1 or give a valid total number")
        else:
            if num > total:
                raise AttributeError("Num has to be smaller than total")

        if name is None:
            name = "progress"

        x = np.asarray([num, total - num])

        win = self.vis.pie(
            X=x,
            opts=dict(legend=['Done', 'To-Do'], title="Progress"),
            win=name,
            env=self.name + env_appendix
        )

        return win

    @convert_params
    def show_histogram(self, array, name=None, bins=30, env_appendix=""):
        """
        Displays the histogramm of an array.

        :param array: The array the histogram is calculated of
        :param name: The name of the window
        :param bins: Number of bins (== bars) in the histogram
        :param env_appendix: appendix to the environment name, if used the new env is env+env_appendix
        """

        vis_task = {
            "type": "histogram",
            "array": array,
            "bins": bins,
            "name": name,
            "env_appendix": env_appendix
        }
        self._queue.put_nowait(vis_task)

    def __show_histogram(self, array, name=None, bins=30, env_appendix="", **kwargs):
        """
       Internal show_histogram method, called by the internal process.
       This function does all the magic.
        """

        win = self.vis.histogram(
            X=array,
            win=name,
            env=self.name + env_appendix,
            opts=dict(numbins=bins, title=name)
        )

        return win

    @convert_params
    def show_histogram_3d(self, array, name, bins=50, env_appendix=""):
        """
        Displays a history of histograms as consequtive lines in a 3d space (similar to tensorflow)

        :param array: New sample of the array that should be plotted (i.e. results in one line of the 3d histogramm)
        :param name: The name of the window
        :param bins: Number of bins in the histogram
        :param env_appendix: appendix to the environment name, if used the new env is env+env_appendix
        """

        vis_task = {
            "type": "histogram_3d",
            "array": array,
            "bins": bins,
            "name": name,
            "env_appendix": env_appendix
        }
        self._queue.put_nowait(vis_task)

    def __show_histogram_3d(self, array, name, bins=50, env_appendix="", **kwargs):
        """
       Internal show_histogram_3d method, called by the internal process.
       This function does all the magic.
        """

        if name not in self._3d_histograms:
            self._3d_histograms[name] = []

        self._3d_histograms[name].append(array)

        if len(self._3d_histograms[name]) > 50:

            n_histo = len(self._3d_histograms[name]) - 20
            every_n = n_histo // 30 + 1

            X = self._3d_histograms[name][:-20][0::every_n] + self._3d_histograms[name][-20:]
        else:
            X = self._3d_histograms[name]

        win = self.vis.histogram_3d(
            X=X,
            win=name,
            env=self.name + env_appendix,
            opts={"numbins": bins, "title": name}
        )

        return win

    @convert_params
    def show_barplot(self, array, legend=None, rownames=None, name=None, env_appendix=""):
        """
        Displays a bar plot from an array

        :param array: array of shape NxM where N is the nomber of rows and M is the number of elements in the row.
        :param legend: list of legend names. Has to have M elements.
        :param rownames: list of row names. Has to have N elements.
        :param name: The name of the window
        :param env_appendix: appendix to the environment name, if used the new env is env+env_appendix
        """

        vis_task = {
            "type": "barplot",
            "array": array,
            "legend": legend,
            "rownames": rownames,
            "name": name,
            "env_appendix": env_appendix
        }
        self._queue.put_nowait(vis_task)

    def __show_barplot(self, array, legend=None, rownames=None, name=None, env_appendix="", **kwargs):
        """
       Internal show_barplot method, called by the internal process.
       This function does all the magic.
        """

        win = self.vis.bar(
            X=array,
            win=name,
            env=self.name + env_appendix,
            opts=dict(
                stacked=False,
                legend=legend,
                rownames=rownames,
                title=name
            )
        )

        return win

    @convert_params
    def show_lineplot(self, y_vals, x_vals=None, name=None, env_appendix=""):
        """
        Displays (multiple) lines plot, given values Y (and optional the corresponding X values)

        :param y_vals: Array of shape MxN , where M is the number of points and N is the number of different line
        :param x_vals: Has to have the same shape as Y: MxN. For each point in Y it gives the corresponding X value (if
         not set the points are assumed to be equally distributed in the interval [0, 1] )
        :param name: The name of the window
        :param env_appendix: appendix to the environment name, if used the new env is env+env_appendix
        """

        vis_task = {
            "type": "lineplot",
            "x_vals": x_vals,
            "y_vals": y_vals,
            "name": name,
            "env_appendix": env_appendix
        }
        self._queue.put_nowait(vis_task)

    def __show_lineplot(self, y_vals, x_vals=None, name=None, env_appendix="", **kwargs):
        """
       Internal show_lineplot method, called by the internal process.
       This function does all the magic.
        """

        win = self.vis.line(
            X=x_vals,
            Y=y_vals,
            win=name,
            env=self.name + env_appendix,
            opts=dict(
                title=name
            )
        )

        return win

    @convert_params
    def show_scatterplot(self, array, labels=None, name=None, env_appendix=""):
        """
        Displays a scatter plots, with the points given in X

        :param array: A 2d array with size N x dim, where each element i \in N at X[i] results in a a 2d (if dim = 2)/
         3d (if dim = 3) point.
        :param labels: For each point a int label (starting from 1) can be given. Has to be an array of size N.
        :param name: The name of the window
        :param env_appendix: appendix to the environment name, if used the new env is env+env_appendix
        """

        vis_task = {
            "type": "scatterplot",
            "array": array,
            "labels": labels,
            "name": name,
            "env_appendix": env_appendix
        }
        self._queue.put_nowait(vis_task)

    def __show_scatterplot(self, array, labels=None, name=None, env_appendix="", **kwargs):
        """
       Internal show_scatterplot method, called by the internal process.
       This function does all the magic.
        """

        win = self.vis.scatter(
            X=array,
            Y=labels,
            win=name,
            env=self.name + env_appendix,
            opts=dict(
                title=name
            )
        )

        return win

    @convert_params
    def show_piechart(self, array, name=None, env_appendix=""):
        """
        Displays a pie chart.

        :param array: Array of positive integers. Each integer will be presented as a part of the pie (with the total
        as the sum of all integers)
        :param name: The name of the window
        :param env_appendix: appendix to the environment name, if used the new env is env+env_appendix
        """

        vis_task = {
            "type": "piechart",
            "array": array,
            "name": name,
            "env_appendix": env_appendix
        }
        self._queue.put_nowait(vis_task)

    def __show_piechart(self, array, name=None, env_appendix="", **kwargs):
        """
       Internal show_piechart method, called by the internal process.
       This function does all the magic.
        """

        win = self.vis.pie(
            X=array,
            win=name,
            env=self.name + env_appendix,
            opts=dict(
                title=name
            )
        )

        return win

    def show_values(self, val_dict):
        """A util function for multiple values. Simple plots all values in a dict, there the window name is the key
        in the dict and the plotted value is the value of a dict (Simply calls the show_value function).

        :param val_dict: Dict with key, values pairs which will be plotted
        """

        for name, value in val_dict:
            self.show_value(value, name)

    def close_all(self):
        """Closes all visdom windows."""
        self.vis.close()

    def exit(self):
        """Kills the internal process."""
        if self._process is not None:
            self._process.terminate()
