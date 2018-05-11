from __future__ import print_function

import datetime
import logging
import os
import sys

from vislogger.logger.abstractlogger import AbstractLogger
from vislogger.util import random_string


class TextLogger(AbstractLogger):
    """A single class for logging"""

    def __init__(self,
                 base_dir=None,
                 logging_level=logging.DEBUG,
                 logging_stream=sys.stdout,
                 default_stream_handler=True,
                 **kwargs):

        super(TextLogger, self).__init__(**kwargs)

        self.base_dir = base_dir
        self.logging_level = logging_level
        self.logging_stream = logging_stream
        self.loggers = dict()
        self.stream_handler_formatter = logging.Formatter('%(message)s')
        self.file_handler_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

        self.init_time = datetime.datetime.today()

        # set up logging
        self.logging_identifier = random_string(10)
        self.add_logger("default", stream_handler=default_stream_handler)

    def add_logger(self, name, logging_level=None, file_handler=True, stream_handler=True):

        self.loggers[name] = logging.getLogger(name + "-" + self.logging_identifier)

        if logging_level is None:
            self.loggers[name].setLevel(self.logging_level)
        else:
            self.loggers[name].setLevel(logging_level)

        if file_handler is True:
            self.add_file_handler(name, name)
        elif isinstance(file_handler, (list, tuple, set)):
            for fh in file_handler:
                if isinstance(fh, logging.FileHandler):
                    self.add_handler(fh, name)
                else:
                    self.add_file_handler(fh, name)
        elif isinstance(file_handler, logging.FileHandler):
            self.add_handler(file_handler, name)
        else:
            pass

        if stream_handler:
            self.add_stream_handler(name)

    def add_handler(self, handler, logger="default"):
        self.loggers[logger].addHandler(handler)

    def add_file_handler(self, name, logger="default"):
        file_handler = logging.FileHandler(os.path.join(self.base_dir, name + ".log"))
        file_handler.setFormatter(self.file_handler_formatter)
        self.add_handler(file_handler, logger)

    def add_stream_handler(self, logger="default"):
        stream_handler = logging.StreamHandler(self.logging_stream)
        stream_handler.setFormatter(self.stream_handler_formatter)
        self.add_handler(stream_handler, logger)

    def print(self, *args, logger="default"):
        """Prints and logs an object"""
        self.log(" ".join(map(str, args)), logger)

    def log(self, msg, logger="default"):
        """Logs a string as info log"""
        self.loggers[logger].info(msg)

    def info(self, msg, logger="default"):
        """wrapper for logger.info"""
        self.loggers[logger].info(msg)

    def debug(self, msg, logger="default"):
        """wrapper for logger.debug"""
        self.loggers[logger].debug(msg)

    def error(self, msg, logger="default"):
        """wrapper for logger.error"""
        self.loggers[logger].error(msg)

    def log_to(self, msg, name, log_level=logging.INFO, log_to_default=False):
        """Logs to an existing logger or creates new one"""

        if name not in self.loggers:
            self.add_logger(name)
        self.log(msg, name)
        if log_to_default:
            self.log(msg, "default")

    def show_text(self, text, name=None, logger="default", **kwargs):
        if name is not None:
            self.log("{}: {}".format(name, text), logger)
        else:
            self.log(text, logger)

    def show_value(self, value, name=None, logger="default", **kwargs):
        if name is not None:
            self.log("{}: {}".format(name, value), logger)
        else:
            self.log(value, logger)
