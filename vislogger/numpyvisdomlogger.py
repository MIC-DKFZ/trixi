from __future__ import division, print_function

import atexit
import multiprocessing as mp
import sys
import traceback

import numpy as np

from abstractvisuallogger import AbstractVisualLogger, convert_params
from extravisdom import ExtraVisdom


class NumpyVisdomLogger(AbstractVisualLogger):
    def __init__(self, name, server="http://localhost", port=8097, auto_close=False, **kwargs):
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

            except:
                
                error = sys.exc_info()[0]
                msg = traceback.format_exc()
                print("Error {}: {}".format(error, msg))

    @convert_params
    def show_image(self, image, name=None, title=None, caption=None, env_app=""):

        vis_task = {
            "type": "image",
            "image": image,
            "name": name,
            "title": title,
            "caption": caption,
            "env_app": env_app
        }
        self._queue.put_nowait(vis_task)

    def __show_image(self, image, name=None, title=None, caption=None, env_app="", **kwargs):

        win = self.vis.image(
            img=image,
            win=name,
            env=self.name + env_app,
            opts=dict(title=title, caption=caption)
        )

        return win

    @convert_params
    def show_images(self, images, name=None, title=None, caption=None, env_app=""):

        vis_task = {
            "type": "images",
            "images": images,
            "name": name,
            "title": title,
            "caption": caption,
            "env_app": env_app
        }
        self._queue.put_nowait(vis_task)

    def __show_images(self, images, name=None, title=None, caption=None, env_app="", **kwargs):

        win = self.vis.images(
            tensor=images,
            win=name,
            env=self.name + env_app,
            opts=dict(title=title, caption=caption)
        )

        return win

    @convert_params
    def show_value(self, value, name=None, env_app=""):
        """Creates a line plot that is automatically appended with new values."""

        vis_task = {
            "type": "value",
            "value": value,
            "name": name,
            "env_app": env_app
        }
        self._queue.put_nowait(vis_task)

    def __show_value(self, value, name=None, env_app="", **kwargs):

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
            env=self.name + env_app,
            opts=dict(title=name)
        )

        return win

    @convert_params
    def show_text(self, text, name=None, title=None, env_app=""):

        vis_task = {
            "type": "text",
            "text": text,
            "name": name,
            "title": title,
            "env_app": env_app
        }
        self._queue.put_nowait(vis_task)

    def __show_text(self, text, name=None, title=None, env_app="", **kwargs):

        win = self.vis.text(
            text=text,
            win=name,
            env=self.name + env_app,
            opts=dict(title=title)
        )

        return win

    @convert_params
    def show_progress(self, num, total=None, name=None, env_app=""):

        vis_task = {
            "type": "progress",
            "num": num,
            "total": total,
            "name": name,
            "env_app": env_app
        }
        self._queue.put_nowait(vis_task)

    def __show_progress(self, num, total=None, name=None, env_app="", **kwargs):

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
            env=self.name + env_app
        )

        return win

    @convert_params
    def show_histogram(self, array, name=None, bins=30, env_app=""):

        vis_task = {
            "type": "histogram",
            "array": array,
            "bins": bins,
            "name": name,
            "env_app": env_app
        }
        self._queue.put_nowait(vis_task)

    def __show_histogram(self, array, name=None, bins=30, env_app="", **kwargs):

        win = self.vis.histogram(
            X=array,
            win=name,
            env=self.name + env_app,
            opts=dict(numbins=bins, title=name)
        )

        return win

    @convert_params
    def show_histogram_3d(self, array, name, bins=50, env_app=""):

        vis_task = {
            "type": "histogram_3d",
            "array": array,
            "bins": bins,
            "name": name,
            "env_app": env_app
        }
        self._queue.put_nowait(vis_task)

    def __show_histogram_3d(self, array, name, bins=50, env_app="", **kwargs):

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
            env=self.name + env_app,
            opts={"numbins": bins, "title": name}
        )

        return win

    @convert_params
    def show_barplot(self, array, legend=None, rownames=None, name=None, env_app=""):

        vis_task = {
            "type": "barplot",
            "array": array,
            "legend": legend,
            "rownames": rownames,
            "name": name,
            "env_app": env_app
        }
        self._queue.put_nowait(vis_task)

    def __show_barplot(self, array, legend=None, rownames=None, name=None, env_app="", **kwargs):

        win = self.vis.bar(
            X=array,
            win=name,
            env=self.name + env_app,
            opts=dict(
                stacked=False,
                legend=legend,
                rownames=rownames,
                title=name
            )
        )

        return win

    @convert_params
    def show_lineplot(self, *args, **kwargs):
        raise NotImplementedError()

    @convert_params
    def show_scatterplot(self, *args, **kwargs):
        raise NotImplementedError()

    @convert_params
    def show_piechart(self, *args, **kwargs):
        raise NotImplementedError()

    def show_values(self, val_dict):
        for name, value in val_dict:
            self.show_value(value, name)

    def close_all(self):
        self.vis.close()

    def exit(self):
        if self._process is not None:
            self._process.terminate()
