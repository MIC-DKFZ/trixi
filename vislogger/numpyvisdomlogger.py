from __future__ import division, print_function

from collections import defaultdict

import atexit
import multiprocessing as mp
import sys
import traceback

import numpy as np

from vislogger.abstractlogger import AbstractLogger, convert_params
from vislogger.extravisdom import ExtraVisdom


class NumpyVisdomLogger(AbstractLogger):
    """
    Visual logger, inherits the AbstractLogger and plots/ logs numpy arrays/ values on a Visdom server.
    """

    def __init__(self, name="main", server="http://localhost", port=8080, auto_close=True, auto_start=False,
                 auto_start_ports=(8080, 8000), **kwargs):
        """
        Creates a new NumpyVisdomLogger object.

        :param name: The name of the visdom environment
        :param server: The address of the visdom server
        :param port: The port of the visdom server
        :param auto_close: Close all objects and kill process at the end of the python script
        """
        super(NumpyVisdomLogger, self).__init__(**kwargs)

        if auto_start:
            auto_port = start_visdom(auto_start_ports)
            if auto_port != -1:
                port = auto_port
                server = "http://localhost"

        self.name = name
        self.server = server
        self.port = port

        self.vis = ExtraVisdom(env=self.name, server=self.server, port=self.port)

        self._value_counter = defaultdict(dict)
        self._3d_histograms = dict()

        self._queue = mp.Queue()
        self._process = mp.Process(target=self.__show, args=(self._queue,))

        if auto_close:
            # atexit.register(self.close_all)
            atexit.register(self.exit)
            atexit.register(self.save_vis)

        self._process.start()

    def __show(self, queue):
        """
        Loop for the internal process to process all visualization tasks

        :param queue: queue with all visualization tasks
        """

        while True:
            vis_task = queue.get()

            try:

                show_fn = self.show_funcs[vis_task["type"]]
                show_fn(self, **vis_task)

            except Exception as e:

                error = sys.exc_info()[0]
                msg = traceback.format_exc()
                print("Error {}: {}".format(error, msg))

    @convert_params
    def show_image(self, image, name=None, title=None, caption=None, env_appendix="", opts=None, **kwargs):
        """
        Displays an image in a window/pane at the visdom server

        :param image: The image array to be displayed
        :param name: The name of the image window
        :param title: The title of the image window
        :param caption: The of the image, displayed in the window
        :param env_appendix: appendix to the environment name, if used the new env is env+env_appendix
        :param opts: opts dict for the ploty/ visdom plot, i.e. can set window size, en/disable ticks,...
        """

        if opts is None:
            opts = {}
        vis_task = {
            "type": "image",
            "image": image,
            "name": name,
            "title": title,
            "caption": caption,
            "env_appendix": env_appendix,
            "opts": opts
        }
        self._queue.put_nowait(vis_task)

    def __show_image(self, image, name=None, title=None, caption=None, env_appendix="", opts=None, **kwargs):
        """
        Internal show_image method, called by the internal process.
        This function does all the magic.
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
    def show_images(self, images, name=None, title=None, caption=None, env_appendix="", opts=None, **kwargs):
        """
        Displays multiple images in a window/pane at a visdom server

        :param images: The image array to be displayed
        :param name: The name of the window
        :param title: The title of the image window
        :param caption: The of the image, displayed in the window
        :param env_appendix: appendix to the environment name, if used the new env is env+env_appendix
        :param opts: opts dict for the ploty/ visdom plot, i.e. can set window size, en/disable ticks,...
        """

        if opts is None:
            opts = {}
        vis_task = {
            "type": "images",
            "images": images,
            "name": name,
            "title": title,
            "caption": caption,
            "env_appendix": env_appendix,
            "opts": opts
        }
        self._queue.put_nowait(vis_task)

    def __show_images(self, images, name=None, title=None, caption=None, env_appendix="", opts=None, **kwargs):
        """
       Internal show_images method, called by the internal process.
       This function does all the magic.
        """
        if opts is None:
            opts = {}
        opts = opts.copy()
        opts.update(dict(
            title=title,
            caption=caption
        ))

        win = self.vis.images(
            tensor=images,
            win=name,
            env=self.name + env_appendix,
            opts=opts
        )

        return win

    @convert_params
    def show_value(self, value, name=None, count=None, tag=None, show_legend=True, env_appendix="", opts=None,
                   **kwargs):
        """
        Creates a line plot that is automatically appended with new values.

        :param value: Value to be plotted / appended to the graph
        :param name: The name of the window
        :param env_appendix: appendix to the environment name, if used the new env is env+env_appendix
        :param opts: opts dict for the ploty/ visdom plot, i.e. can set window size, en/disable ticks,...
        """

        if opts is None:
            opts = {}
        vis_task = {
            "type": "value",
            "value": value,
            "name": name,
            "count": count,
            "tag": tag,
            "show_legend": show_legend,
            "env_appendix": env_appendix,
            "opts": opts
        }
        self._queue.put_nowait(vis_task)

    def __show_value(self, value, name=None, count=None, tag=None, show_legend=True,  env_appendix="", opts=None, **kwargs):
        """
       Internal show_value method, called by the internal process.
       This function does all the magic.
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
        if name is not None and name in self._value_counter:
            up_str = "append"

        if name is not None and name in self._value_counter and tag in self._value_counter[name]:
            if count is None:
                self._value_counter[name][tag] += 1
            else:
                self._value_counter[name][tag] += count - self._value_counter[name][tag]
        else:
            if count is None:
                count = 0
            if value_dim == 1:
                self._value_counter[name][tag] = np.array([count])
            else:
                self._value_counter[name][tag] = np.array([[count] * value_dim])

        opts = opts.copy()
        opts.update(dict(
            title=name,
        ))

        win = self.vis.line(
            Y=value,
            X=self._value_counter[name][tag],
            win=name,
            name=tag,
            update=up_str,
            env=self.name + env_appendix,
            opts=opts
        )

        return win

    @convert_params
    def show_text(self, text, name=None, env_appendix="", opts=None, **kwargs):
        """
        Displays a text in a visdom window

        :param text: The text to be displayed
        :param name: The name of the window
        :param env_appendix: appendix to the environment name, if used the new env is env+env_appendix
        :param opts: opts dict for the ploty/ visdom plot, i.e. can set window size, en/disable ticks,...
        """

        if opts is None:
            opts = {}
        vis_task = {
            "type": "text",
            "text": text,
            "name": name,
            "env_appendix": env_appendix,
            "opts": opts
        }
        self._queue.put_nowait(vis_task)

    def __show_text(self, text, name=None, env_appendix="", opts=None, **kwargs):
        """
       Internal show_text method, called by the internal process.
       This function does all the magic.
        """

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
    def show_progress(self, num, total=None, name=None, env_appendix="", opts=None, **kwargs):
        """
        Shows the progress as a pie chart.

        :param num: Current progress. Either a relative value (0 <= num <= 1) or a absolute value if total is given (
        but has to be smaller than total)
        :param total: This if the total number of iterations.
        :param name: The name of the window
        :param env_appendix: appendix to the environment name, if used the new env is env+env_appendix
        :param opts: opts dict for the ploty/ visdom plot, i.e. can set window size, en/disable ticks,...
        """

        if opts is None:
            opts = {}
        vis_task = {
            "type": "progress",
            "num": num,
            "total": total,
            "name": name,
            "env_appendix": env_appendix,
            "opts": opts
        }
        self._queue.put_nowait(vis_task)

    def __show_progress(self, num, total=None, name=None, env_appendix="", opts=None, **kwargs):
        """
       Internal show_progress method, called by the internal process.
       This function does all the magic.
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
    def show_histogram(self, array, name=None, bins=30, env_appendix="", opts=None, **kwargs):
        """
        Displays the histogramm of an array.

        :param array: The array the histogram is calculated of
        :param name: The name of the window
        :param bins: Number of bins (== bars) in the histogram
        :param env_appendix: appendix to the environment name, if used the new env is env+env_appendix
        :param opts: opts dict for the ploty/ visdom plot, i.e. can set window size, en/disable ticks,...
        """

        if opts is None:
            opts = {}
        vis_task = {
            "type": "histogram",
            "array": array,
            "bins": bins,
            "name": name,
            "env_appendix": env_appendix,
            "opts": opts
        }
        self._queue.put_nowait(vis_task)

    def __show_histogram(self, array, name=None, bins=30, env_appendix="", opts=None, **kwargs):
        """
       Internal show_histogram method, called by the internal process.
       This function does all the magic.
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
    def show_histogram_3d(self, array, name, bins=50, env_appendix="", opts=None, **kwargs):
        """
        Displays a history of histograms as consequtive lines in a 3d space (similar to tensorflow)

        :param array: New sample of the array that should be plotted (i.e. results in one line of the 3d histogramm)
        :param name: The name of the window
        :param bins: Number of bins in the histogram
        :param env_appendix: appendix to the environment name, if used the new env is env+env_appendix
        :param opts: opts dict for the ploty/ visdom plot, i.e. can set window size, en/disable ticks,...
        """

        if opts is None:
            opts = {}
        vis_task = {
            "type": "histogram_3d",
            "array": array,
            "bins": bins,
            "name": name,
            "env_appendix": env_appendix,
            "opts": opts
        }
        self._queue.put_nowait(vis_task)

    def __show_histogram_3d(self, array, name, bins=50, env_appendix="", opts=None, **kwargs):
        """
       Internal show_histogram_3d method, called by the internal process.
       This function does all the magic.
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
    def show_barplot(self, array, legend=None, rownames=None, name=None, env_appendix="", opts=None, **kwargs):
        """
        Displays a bar plot from an array

        :param array: array of shape NxM where N is the nomber of rows and M is the number of elements in the row.
        :param legend: list of legend names. Has to have M elements.
        :param rownames: list of row names. Has to have N elements.
        :param name: The name of the window
        :param env_appendix: appendix to the environment name, if used the new env is env+env_appendix
        :param opts: opts dict for the ploty/ visdom plot, i.e. can set window size, en/disable ticks,...
        """

        if opts is None:
            opts = {}
        vis_task = {
            "type": "barplot",
            "array": array,
            "legend": legend,
            "rownames": rownames,
            "name": name,
            "env_appendix": env_appendix,
            "opts": opts
        }
        self._queue.put_nowait(vis_task)

    def __show_barplot(self, array, legend=None, rownames=None, name=None, env_appendix="", opts=None, **kwargs):
        """
       Internal show_barplot method, called by the internal process.
       This function does all the magic.
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
    def show_lineplot(self, y_vals, x_vals=None, name=None, env_appendix="", opts=None, **kwargs):
        """
        Displays (multiple) lines plot, given values Y (and optional the corresponding X values)

        :param y_vals: Array of shape MxN , where M is the number of points and N is the number of different line
        :param x_vals: Has to have the same shape as Y: MxN. For each point in Y it gives the corresponding X value (if
         not set the points are assumed to be equally distributed in the interval [0, 1] )
        :param name: The name of the window
        :param env_appendix: appendix to the environment name, if used the new env is env+env_appendix
        :param opts: opts dict for the ploty/ visdom plot, i.e. can set window size, en/disable ticks,...
        """

        if opts is None:
            opts = {}
        vis_task = {
            "type": "lineplot",
            "x_vals": x_vals,
            "y_vals": y_vals,
            "name": name,
            "env_appendix": env_appendix,
            "opts": opts
        }
        self._queue.put_nowait(vis_task)

    def __show_lineplot(self, y_vals, x_vals=None, name=None, env_appendix="", opts=None, **kwargs):
        """
       Internal show_lineplot method, called by the internal process.
       This function does all the magic.
        """

        if opts is None:
            opts = {}
        opts = opts.copy()
        opts.update(dict(
            title=name,
        ))

        win = self.vis.line(
            X=x_vals,
            Y=y_vals,
            win=name,
            env=self.name + env_appendix,
            opts=opts
        )

        return win

    @convert_params
    def show_scatterplot(self, array, labels=None, name=None, env_appendix="", opts=None, **kwargs):
        """
        Displays a scatter plots, with the points given in X

        :param array: A 2d array with size N x dim, where each element i \in N at X[i] results in a a 2d (if dim = 2)/
         3d (if dim = 3) point.
        :param labels: For each point a int label (starting from 1) can be given. Has to be an array of size N.
        :param name: The name of the window
        :param env_appendix: appendix to the environment name, if used the new env is env+env_appendix
        :param opts: opts dict for the ploty/ visdom plot, i.e. can set window size, en/disable ticks,...
        """

        if opts is None:
            opts = {}
        vis_task = {
            "type": "scatterplot",
            "array": array,
            "labels": labels,
            "name": name,
            "env_appendix": env_appendix,
            "opts": opts
        }
        self._queue.put_nowait(vis_task)

    def __show_scatterplot(self, array, labels=None, name=None, env_appendix="", opts=None, **kwargs):
        """
       Internal show_scatterplot method, called by the internal process.
       This function does all the magic.
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
    def show_piechart(self, array, name=None, env_appendix="", opts=None, **kwargs):
        """
        Displays a pie chart.

        :param array: Array of positive integers. Each integer will be presented as a part of the pie (with the total
        as the sum of all integers)
        :param name: The name of the window
        :param env_appendix: appendix to the environment name, if used the new env is env+env_appendix
        :param opts: opts dict for the ploty/ visdom plot, i.e. can set window size, en/disable ticks,...
        """

        if opts is None:
            opts = {}
        vis_task = {
            "type": "piechart",
            "array": array,
            "name": name,
            "env_appendix": env_appendix,
            "opts": opts
        }
        self._queue.put_nowait(vis_task)

    def __show_piechart(self, array, name=None, env_appendix="", opts=None, **kwargs):
        """
       Internal show_piechart method, called by the internal process.
       This function does all the magic.
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
    def show_svg(self, svg, name=None, env_appendix="", opts=None, **kwargs):
        """
        Displays a svg file.

        :param svg: Filename of the svg file which should be displayed
        :param name: The name of the window
        :param env_appendix: appendix to the environment name, if used the new env is env+env_appendix
        :param opts: opts dict for the ploty/ visdom plot, i.e. can set window size, en/disable ticks,...
        """

        if opts is None:
            opts = {}
        vis_task = {
            "type": "svg",
            "svg": svg,
            "name": name,
            "env_appendix": env_appendix,
            "opts": opts
        }
        self._queue.put_nowait(vis_task)

    def __show_svg(self, svg, name=None, env_appendix="", opts=None, **kwargs):
        """
       Internal show_svg method, called by the internal process.
       This function does all the magic.
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

        :param val_dict: Dict with key, values pairs which will be plotted
        """

        for name, value in val_dict:
            self.show_value(value, name)

    @convert_params
    def add_to_graph(self, y_vals, x_vals=None, name=None, legend_name=None, append=True, env_appendix="", opts=None,
                     **kwargs):
        """
        Displays (multiple) lines plot, given values Y (and optional the corresponding X values)

        :param y_vals: Array of shape MxN , where M is the number of points and N is the number of different line
        :param x_vals: Has to have the same shape as Y: MxN. For each point in Y it gives the corresponding X value (if
         not set the points are assumed to be equally distributed in the interval [0, 1] )
        :param name: The name of the window
        :param env_appendix: appendix to the environment name, if used the new env is env+env_appendix
        :param opts: opts dict for the ploty/ visdom plot, i.e. can set window size, en/disable ticks,...
        """

        if opts is None:
            opts = {}
        vis_task = {
            "type": "add",
            "x_vals": x_vals,
            "y_vals": y_vals,
            "name": name,
            "legend_name": legend_name,
            "append": append,
            "env_appendix": env_appendix,
            "opts": opts
        }
        self._queue.put_nowait(vis_task)

    def __add_to_graph(self, y_vals, x_vals=None, name=None, legend_name=None, append=True, env_appendix="", opts=None,
                       **kwargs):
        """
       Internal show_lineplot method, called by the internal process.
       This function does all the magic.
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
    def send_data(self, data, name=None, layout=None, endpoint='events', append=False, **kwargs):
        """
        Displays (multiple) lines plot, given values Y (and optional the corresponding X values)

        :param y_vals: Array of shape MxN , where M is the number of points and N is the number of different line
        :param x_vals: Has to have the same shape as Y: MxN. For each point in Y it gives the corresponding X value (if
         not set the points are assumed to be equally distributed in the interval [0, 1] )
        :param name: The name of the window
        :param env_appendix: appendix to the environment name, if used the new env is env+env_appendix
        :param opts: opts dict for the ploty/ visdom plot, i.e. can set window size, en/disable ticks,...
        """

        if layout is None:
            layout = {}
        vis_task = {
            "type": "data",
            "data": data,
            "layout": layout,
            "name": name,
            "endpoint": endpoint,
            "append": append
        }
        self._queue.put_nowait(vis_task)

    def __send_data(self, data, name=None, layout=None, endpoint='events', append=False, **kwargs):
        """
       Internal show_lineplot method, called by the internal process.
       This function does all the magic.
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

    show_funcs = {
        "image": __show_image,
        "images": __show_images,
        "value": __show_value,
        "text": __show_text,
        "progress": __show_progress,
        "histogram": __show_histogram,
        "histogram_3d": __show_histogram_3d,
        "barplot": __show_barplot,
        "lineplot": __show_lineplot,
        "scatterplot": __show_scatterplot,
        "piechart": __show_piechart,
        "svg": __show_svg,
        "add": __add_to_graph,
        "data": __send_data,
    }


def start_visdom(port_list=(8080, 8000)):
    import time
    from multiprocessing import Process
    import visdom.server

    from vislogger.util import PyLock

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
        p = Process(target=visdom.server.start_server, kwargs={"port": port})
        atexit.register(p.terminate)
        p.start()
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
                    print("Started Visdom on Port:", port)
                    lock.__exit__(0, 0, 0)
                    return port
            else:
                i += 1
        except Exception:
            pass

    return -1
