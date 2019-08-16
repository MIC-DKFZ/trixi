from __future__ import division, print_function

import atexit
from collections import defaultdict
import inspect

import os

if os.name == "nt":
    IS_WINDOWS = True
    from queue import Queue
    from threading import Thread as Process
else:
    IS_WINDOWS = False
    from pathos.helpers import mp
    Queue = mp.Queue
    Process = mp.Process
import sys
import traceback

import numpy as np

from trixi.logger.abstractlogger import AbstractLogger, convert_params
from trixi.util import ExtraVisdom


def add_to_queue(func):
    def wrapper(self, *args, **kwargs):
        tpl = (func, args, kwargs)
        self._queue.put_nowait(tpl)

    return wrapper


class NumpyVisdomLogger(AbstractLogger):
    """
    Visual logger, inherits the AbstractLogger and plots/ logs numpy arrays/ values on a Visdom server.
    """

    def __init__(self, exp_name="main", server="http://localhost", port=8080, auto_close=True, auto_start=False,
                 auto_start_ports=(8080, 8000), **kwargs):
        """
        Creates a new NumpyVisdomLogger object.

        Args:
            name: The name of the visdom environment
            server: The address of the visdom server
            port: The port of the visdom server
            auto_close: Close all objects and kill process at the end of the python script
            auto_start: Flag, if it should try to start a visdom server on the given ports
            auto_start_ports: Ordered list of ports, to try to start a visdom server on (only on the first available
            port)
        """
        super(NumpyVisdomLogger, self).__init__(**kwargs)

        if auto_start:
            auto_port = start_visdom(auto_start_ports)
            if auto_port != -1:
                port = auto_port
                server = "http://localhost"

        self.name = exp_name
        self.server = server
        self.port = port

        self.vis = ExtraVisdom(env=self.name, server=self.server, port=self.port)

        self._value_counter = defaultdict(dict)
        self._3d_histograms = dict()

        self._queue = Queue()
        self._process = Process(target=self.__show, args=(self._queue,))

        if auto_close:
            # atexit.register(self.close_all)
            if not IS_WINDOWS:
                atexit.register(self.exit)
            atexit.register(self.save_vis)

        self._process.start()

    def __show(self, queue):
        """
        Loop for the internal process to process all visualization tasks

        Args:
            queue: queue with all visualization tasks
        """

        while True:

            func, args, kwargs = queue.get()

            try:

                func(self, *args, **kwargs)

            except Exception as e:

                error = sys.exc_info()[0]
                msg = traceback.format_exc()
                print("Error {}: {}".format(error, msg))

    @convert_params
    @add_to_queue
    def show_image(self, image, name=None, title=None, caption=None, env_appendix="", opts=None, **kwargs):
        """
        Displays an image in a window/pane at the visdom server

        Args:
            image: The image array to be displayed
            name: The name of the image window
            title: The title of the image window
            caption: The of the image, displayed in the window
            env_appendix: appendix to the environment name, if used the new env is env+env_appendix
            opts: opts dict for the ploty/ visdom plot, i.e. can set window size, en/disable ticks,...
        """

        if opts is None:
            opts = {}
        opts = opts.copy()
        opts.update(dict(
            title=title,
            caption=caption
        ))

        win = self.vis.image(
            img=image,
            win=name,
            env=self.name + env_appendix,
            opts=opts
        )

        return win

    @convert_params
    @add_to_queue
    def show_images(self, images, name=None, title=None, caption=None, env_appendix="", opts=None, **kwargs):
        """
        Displays multiple images in a window/pane at a visdom server

        Args:
            images: The image array to be displayed
            name: The name of the window
            title: The title of the image window
            caption: The of the image, displayed in the window
            env_appendix: appendix to the environment name, if used the new env is env+env_appendix
            opts: opts dict for the ploty/ visdom plot, i.e. can set window size, en/disable ticks,...
        """
        if opts is None:
            opts = {}
        else:
            if 'nrow' in opts.keys():
                nrow = opts['nrow']
            else:
                nrow = 8  # as defined in function
        opts = opts.copy()
        opts.update(dict(
            title=title,
            caption=caption
        ))

        win = self.vis.images(
            tensor=images,
            win=name,
            env=self.name + env_appendix,
            opts=opts,
            # nrow=nrow
        )

        return win

    @convert_params
    @add_to_queue
    def show_value(self, value, name=None, counter=None, tag=None, show_legend=True, env_appendix="", opts=None,
                     **kwargs):
        """
        Creates a line plot that is automatically appended with new values and plots it to a visdom server.

        Args:
            value: Value to be plotted / appended to the graph (y-axis value)
            name: The name of the window
            counter: counter, which tells the number of the sample (with the same name) (x-axis value)
            tag: Tag, grouping similar values. Values with the same tag will be plotted in the same plot
            show_legend (bool): Flag, if it should display a legend
            env_appendix: appendix to the environment name, if used the new env is env+env_appendix
            opts: opts dict for the ploty/ visdom plot, i.e. can set window size, en/disable ticks,...
        """

        if opts is None:
            opts = {}
        value_dim = 1
        if isinstance(value, (list, tuple, np.ndarray)):
            value_dim = len(value)
            value = np.asarray([value])
        else:
            value = np.asarray([value])
        if tag is None:
            tag = ""
            if name is not None:
                tag = name
        if "showlegend" not in opts and show_legend:
            opts["showlegend"] = True

        up_str = None
        if tag is not None and tag in self._value_counter:
            up_str = "append"

        if tag is not None and tag in self._value_counter and name in self._value_counter[tag]:
            if counter is None:
                self._value_counter[tag][name] += 1
            else:
                self._value_counter[tag][name] += counter - self._value_counter[tag][name]
        else:
            if counter is None:
                counter = 0
            if value_dim == 1:
                self._value_counter[tag][name] = np.array([counter])
            else:
                self._value_counter[tag][name] = np.array([[counter] * value_dim])

        opts = opts.copy()
        opts.update(dict(
            title=tag,
        ))

        display_name = tag
        if value_dim != 1:
            display_name = None

        win = self.vis.line(
            Y=value,
            X=self._value_counter[tag][name],
            win=tag,
            name=name,
            update=up_str,
            env=self.name + env_appendix,
            opts=opts
        )

        return win

    @convert_params
    @add_to_queue
    def show_text(self, text, name=None, env_appendix="", opts=None, **kwargs):
        """
        Displays a text in a visdom window

        Args:
             text: The text to be displayed
             name: The name of the window
             env_appendix: appendix to the environment name, if used the new env is env+env_appendix
             opts: opts dict for the ploty/ visdom plot, i.e. can set window size, en/disable ticks,...
        """

        text = text.replace("\n", "<br>")

        if opts is None:
            opts = {}
        win = self.vis.text(
            text=text,
            win=name,
            env=self.name + env_appendix,
            opts=opts
        )

        return win

    @convert_params
    @add_to_queue
    def show_progress(self, num, total=None, name=None, env_appendix="", opts=None, **kwargs):
        """
        Shows the progress as a pie chart.

        Args:
            num: Current progress. Either a relative value (0 <= num <= 1) or a absolute value if total is given (
            but has to be smaller than total)
            total: This if the total number of iterations.
            name: The name of the window
            env_appendix: appendix to the environment name, if used the new env is env+env_appendix
            opts: opts dict for the ploty/ visdom plot, i.e. can set window size, en/disable ticks,...
        """

        if opts is None:
            opts = {}
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

        opts = opts.copy()
        opts.update(dict(
            legend=['Done', 'To-Do'],
            title="Progress")
        )

        win = self.vis.pie(
            X=x,
            win=name,
            env=self.name + env_appendix,
            opts=opts
        )

        return win

    @convert_params
    @add_to_queue
    def show_histogram(self, array, name=None, bins=30, env_appendix="", opts=None, **kwargs):
        """
        Displays the histogramm of an array.

        Args:
            array: The array the histogram is calculated of
            name: The name of the window
            bins: Number of bins (== bars) in the histogram
            env_appendix: appendix to the environment name, if used the new env is env+env_appendix
            opts: opts dict for the ploty/ visdom plot, i.e. can set window size, en/disable ticks,...
        """
        if opts is None:
            opts = {}
        opts = opts.copy()
        opts.update(dict(
            title=name,
            numbins=bins
        ))

        win = self.vis.histogram(
            X=array,
            win=name,
            env=self.name + env_appendix,
            opts=opts
        )

        return win

    @convert_params
    @add_to_queue
    def __show_histogram_3d(self, array, name, bins=50, env_appendix="", opts=None, **kwargs):
        """
        Displays a history of histograms as consequtive lines in a 3d space (similar to tensorflow)

        Args:
            array: New sample of the array that should be plotted (i.e. results in one line of the 3d histogramm)
            name: The name of the window
            bins: Number of bins in the histogram
            env_appendix: appendix to the environment name, if used the new env is env+env_appendix
            opts: opts dict for the ploty/ visdom plot, i.e. can set window size, en/disable ticks,...
        """

        if opts is None:
            opts = {}
        if name not in self._3d_histograms:
            self._3d_histograms[name] = []

        self._3d_histograms[name].append(array)

        if len(self._3d_histograms[name]) > 50:

            n_histo = len(self._3d_histograms[name]) - 20
            every_n = n_histo // 30 + 1

            x = self._3d_histograms[name][:-20][0::every_n] + self._3d_histograms[name][-20:]
        else:
            x = self._3d_histograms[name]

        opts = opts.copy()
        opts.update(dict(
            title=name,
            numbins=bins
        ))

        win = self.vis.histogram_3d(
            X=x,
            win=name,
            env=self.name + env_appendix,
            opts=opts
        )

        return win

    @convert_params
    @add_to_queue
    def show_barplot(self, array, legend=None, rownames=None, name=None, env_appendix="", opts=None, **kwargs):
        """
        Displays a bar plot from an array

        Args:
            array: array of shape NxM where N is the nomber of rows and M is the number of elements in the row.
            legend: list of legend names. Has to have M elements.
            rownames: list of row names. Has to have N elements.
            name: The name of the window
            env_appendix: appendix to the environment name, if used the new env is env+env_appendix
            opts: opts dict for the ploty/ visdom plot, i.e. can set window size, en/disable ticks,...
        """

        if opts is None:
            opts = {}
        opts = opts.copy()
        opts.update(dict(
            stacked=False,
            legend=legend,
            rownames=rownames,
            title=name
        ))

        win = self.vis.bar(
            X=array,
            win=name,
            env=self.name + env_appendix,
            opts=opts
        )

        return win

    @convert_params
    @add_to_queue
    def show_lineplot(self, y_vals, x_vals=None, name=None, env_appendix="", show_legend=True, opts=None, **kwargs):
        """
        Displays (multiple) lines plot, given values Y (and optional the corresponding X values)

        Args:
            y_vals: Array of shape MxN , where M is the number of points and N is the number of different line
            x_vals: Has to have the same shape as Y: MxN. For each point in Y it gives the corresponding X value (if
            not set the points are assumed to be equally distributed in the interval [0, 1] )
            name: The name of the window
            env_appendix: appendix to the environment name, if used the new env is env+env_appendix
            opts: opts dict for the ploty/ visdom plot, i.e. can set window size, en/disable ticks,...
        """

        if opts is None:
            opts = {}
        opts = opts.copy()
        opts.update(dict(
            title=name,
        ))

        if "showlegend" not in opts and show_legend:
            opts["showlegend"] = True

        win = self.vis.line(
            X=x_vals,
            Y=y_vals,
            win=name,
            env=self.name + env_appendix,
            opts=opts
        )

        return win

    @convert_params
    @add_to_queue
    def show_boxplot(self, x_vals, name=None, env_appendix="", opts=None, **kwargs):
        """
        Displays a box plot, given values X

        Args:
            x_vals: Array of shape MxN , where M is the number of points and N are the number of groups
            name: The name of the window
            env_appendix: appendix to the environment name, if used the new env is env+env_appendix
            opts: opts dict for the ploty/ visdom plot, i.e. can set window size, en/disable ticks,...
        """

        if opts is None:
            opts = {}
        opts = opts.copy()
        opts.update(dict(
            title=name,
        ))

        win = self.vis.boxplot(
            X=x_vals,
            win=name,
            env=self.name + env_appendix,
            opts=opts
        )

        return win

    @convert_params
    @add_to_queue
    def show_surfaceplot(self, x_vals, name=None, env_appendix="", opts=None, **kwargs):
        """
        Displays a surface plot given values X

        Args:
            x_vals: Array of shape MxN
            name: The name of the window
            env_appendix: appendix to the environment name, if used the new env is env+env_appendix
            opts: opts dict for the ploty/ visdom plot, i.e. can set window size, en/disable ticks,...
        """

        if opts is None:
            opts = {}
            opts.update(dict(
                title=name,
                colormap='Viridis',
                xlabel='X',
                ylabel='Y'
            ))
        opts = opts.copy()

        win = self.vis.surf(X=x_vals,
                            win=name,
                            env=self.name + env_appendix,
                            opts=opts, )

        return win

    @convert_params
    @add_to_queue
    def show_contourplot(self, x_vals, name=None, env_appendix="", opts=None, **kwargs):
        """
        Displays a contour plot

        Args:
            x_vals: Array of shape MxN
            name: The name of the window
            env_appendix: appendix to the environment name, if used the new env is env+env_appendix
            opts: opts dict for the ploty/ visdom plot, i.e. can set window size, en/disable ticks,...
        """

        if opts is None:
            opts = {}
            opts.update(dict(
                title=name,
                colormap='Viridis',
                xlabel='feature',
                ylabel='batch'
            ))
        opts = opts.copy()

        win = self.vis.contour(X=x_vals,
                               win=name,
                               env=self.name + env_appendix,
                               opts=opts, )

        return win

    @convert_params
    @add_to_queue
    def show_scatterplot(self, array, labels=None, name=None, env_appendix="", opts=None, **kwargs):
        """
        Displays a scatter plots, with the points given in array

        Args:
            array: A 2d array with size N x dim, where each element i \in N at X[i] results in a a 2d (if dim = 2)/
            3d (if dim = 3) point.
            labels: For each point a int label (starting from 1) can be given. Has to be an array of size N.
            name: The name of the window
            env_appendix: appendix to the environment name, if used the new env is env+env_appendix
            opts: opts dict for the ploty/ visdom plot, i.e. can set window size, en/disable ticks,...
        """

        if opts is None:
            opts = {}
        opts.update(dict(
            title=name,
        ))

        win = self.vis.scatter(
            X=array,
            Y=labels,
            win=name,
            env=self.name + env_appendix,
            opts=opts
        )

        return win

    @convert_params
    @add_to_queue
    def show_piechart(self, array, name=None, env_appendix="", opts=None, **kwargs):
        """
        Displays a pie chart.

        Args:
            array: Array of positive integers. Each integer will be presented as a part of the pie (with the total
            as the sum of all integers)
            name: The name of the window
            env_appendix: appendix to the environment name, if used the new env is env+env_appendix
            opts: opts dict for the ploty/ visdom plot, i.e. can set window size, en/disable ticks,...
        """

        if opts is None:
            opts = {}
        opts.update(dict(
            title=name
        ))

        win = self.vis.pie(
            X=array,
            win=name,
            env=self.name + env_appendix,
            opts=opts
        )

        return win

    @convert_params
    @add_to_queue
    def show_svg(self, svg, name=None, env_appendix="", opts=None, **kwargs):
        """
        Displays a svg file.

        Args:
            svg: Filename of the svg file which should be displayed
            name: The name of the window
            env_appendix: appendix to the environment name, if used the new env is env+env_appendix
            opts: opts dict for the ploty/ visdom plot, i.e. can set window size, en/disable ticks,...
        """

        if opts is None:
            opts = {}
        with open(svg, 'r') as fileobj:
            svgstr = fileobj.read()

        opts.update(dict(
            title=name
        ))

        win = self.vis.svg(
            svgstr=svgstr,
            win=name,
            env=self.name + env_appendix,
            opts=opts
        )

        return win

    def show_values(self, val_dict):
        """A util function for multiple values. Simple plots all values in a dict, there the window name is the key
        in the dict and the plotted value is the value of a dict (Simply calls the show_value function).

        Args:
            val_dict: Dict with key, values pairs which will be plotted
        """

        for name, value in val_dict:
            self.show_value(value, name)

    @convert_params
    @add_to_queue
    def add_to_graph(self, y_vals, x_vals=None, name=None, legend_name=None, append=True, env_appendix="", opts=None,
                       **kwargs):
        """
        Displays (multiple) lines plot, given values Y (and optional the corresponding X values)

        Args:
            y_vals: Array of shape MxN , where M is the number of points and N is the number of different line
            x_vals: Has to have the same shape as Y: MxN. For each point in Y it gives the corresponding X value (if
            not set the points are assumed to be equally distributed in the interval [0, 1] )
            name: The name of the window
            env_appendix: appendix to the environment name, if used the new env is env+env_appendix
            opts: opts dict for the ploty/ visdom plot, i.e. can set window size, en/disable ticks,...
        """

        if opts is None:
            opts = {}
        opts.update(dict(
            title=name,
        ))

        win = self.vis.updateTrace(
            X=x_vals,
            Y=y_vals,
            win=name,
            name=legend_name,
            append=append,
            env=self.name + env_appendix,
            opts=opts
        )

        return win

    @convert_params
    @add_to_queue
    def show_matplot_plt(self, plt, name=None, title=None, caption=None, env_appendix="", opts=None, **kwargs):
        """
        Displays an matplotlib figure in a window/pane at the visdom server

        Args:
            plt: The matplotlib figure/plot to be displayed
            name: The name of the image window
            title: The title of the image window
            caption: The of the image, displayed in the window
            env_appendix: appendix to the environment name, if used the new env is env+env_appendix
            opts: opts dict for the ploty/ visdom plot, i.e. can set window size, en/disable ticks,...
        """

        if opts is None:
            opts = {}
        opts = opts.copy()
        opts.update(dict(
            title=title,
            caption=caption
        ))

        win = self.vis.matplot(
            plot=plt,
            win=name,
            env=self.name + env_appendix,
            opts=opts
        )

        return win

    @convert_params
    @add_to_queue
    def show_plotly_plt(self, figure, name=None, env_appendix="", **kwargs):
        """
        Displays an plotly figure in a window/pane at the visdom server

        Args:
            figure: The plotly figure/plot to be displayed
            name: The name of the image window
            title: The title of the image window
            caption: The of the image, displayed in the window
            env_appendix: appendix to the environment name, if used the new env is env+env_appendix
            opts: opts dict for the ploty/ visdom plot, i.e. can set window size, en/disable ticks,...
        """

        win = self.vis.plotlyplot(
            figure=figure,
            win=name,
            env=self.name + env_appendix,
        )

        return win

    @convert_params
    @add_to_queue
    def send_data(self, data, name=None, layout=None, endpoint='events', append=False, **kwargs):
        """
        Sends data to a visdom server.

        Args:
            data: The data to be send (has to be in the plotly / visdom format, see the offical visdom github page)
            name: The name of the window
            layout: Layout of the data
            endpoint: Endpoint to recieve the data (or at least to which visdom endpoint to send it to)
            append: Flag, if data should be appended
        """

        if layout is None:
            layout = {}

        if not isinstance(data, (list, type)):
            data = [data]

        win = self.vis._send({'data': data,
                              'layout': layout,
                              'win': name,
                              'append': append}, endpoint=endpoint)

        return win

    def close_all(self):
        """Closes all visdom windows."""
        self.vis.close()

    def save_vis(self):
        self.vis.save([self.name])

    def exit(self):
        """Kills the internal process."""
        if self._process is not None:
            self._process.terminate()



def start_visdom(port_list=(8080, 8000)):
    """
    Starts a visdom server on a given port

    Args:
        port_list: Priority list of port. Will start a visdom server on the first available port.

    """
    import time
    from multiprocessing import Process
    import visdom.server

    from trixi.util import PyLock

    lock_id = "visdom_lock"

    def is_port_available(port, verbose=False):

        import socket
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            s.bind(("127.0.0.1", port))
        except socket.error as e:
            if e.errno == 98:
                if verbose:
                    print("Port {} is already in use".format(port))
            else:
                if verbose:
                    print(e)
            s.close()
            return False
        s.close()
        return True

    def _start_visdom(port):
        p = Process(target=visdom.server.start_server, kwargs={"port": port, "base_url": ''})
        p.start()
        atexit.register(p.terminate)
        time.sleep(20)
        return True

    lock = PyLock(lock_id, timeout=10)

    i = 0
    while i < len(port_list):
        port = port_list[i]

        try:
            lock.__enter__()
            if is_port_available(port):
                if _start_visdom(port):
                    lock.__exit__(0, 0, 0)
                    print("Started Visdom on Port:", port)
                    return port
            else:
                i += 1
        except Exception:
            pass

    return -1
