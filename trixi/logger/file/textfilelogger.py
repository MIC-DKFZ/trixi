from __future__ import print_function

import datetime
import logging
import os
import sys

from trixi.logger.abstractlogger import AbstractLogger
from trixi.util import random_string


class TextFileLogger(AbstractLogger):
    """
    A Logger for logging text into different text files and output streams (using the python logging framework)
    """

    def __init__(self, base_dir=None, logging_level=logging.DEBUG, logging_stream=sys.stdout,
                 default_stream_handler=True, **kwargs):
        """
        Initializes a TextFileLogger and a default logger

        Args:
            base_dir: Directory to save the log files in
            logging_level: Default logging level for the default log
            logging_stream: default logging level
            default_stream_handler: Falg, if a default stream handler should be added
        """

        super(TextFileLogger, self).__init__(**kwargs)

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
        """
        Adds a new logger

        Args:
            name: Name of the new logger
            logging_level: Logging level of the new logger
            file_handler: Flag, if it should use a file_handler, if yes creates a new file with the given name in the
                logging directory
            stream_handler: Flag, if the logger should also log to the default stream

        Returns:

        """

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
        """
        Adds an additional handler to a logger with a name
        Args:
            handler: Logging handler to be added to a given logger
            logger: Name of the logger to add the hander to

        """
        self.loggers[logger].addHandler(handler)

    def add_file_handler(self, name, logger="default"):
        """
        Adds a file handler to a logger, thus the logger will log into a log file with a given name

        Args:
            name: File name of the log file (in which the logger will log now)
            logger: Name of the logger to add the file-hander/ logging file to

        """
        file_handler = logging.FileHandler(os.path.join(self.base_dir, name + ".log"))
        file_handler.setFormatter(self.file_handler_formatter)
        self.add_handler(file_handler, logger)

    def add_stream_handler(self, logger="default"):
        """
        Adds a stream handler to a logger, thus the logger will log into the default logging stream

        Args:
            logger: Name of the logger to add the stream-hander to

        """
        stream_handler = logging.StreamHandler(self.logging_stream)
        stream_handler.setFormatter(self.stream_handler_formatter)
        self.add_handler(stream_handler, logger)

    def print(self, *args, logger="default"):
        """
        Prints and logs objects

        Args:
            *args: Object to print/ log
            logger: Logger which should log

        """
        self.log(" ".join(map(str, args)), logger)

    def log(self, msg, logger="default"):
        """
        Prints and logs a message with the level info

        Args:
            msg: Message to print/ log
            logger: Logger which should log

        """
        self.loggers[logger].info(msg)

    def info(self, msg, logger="default"):
        """
        Prints and logs a message with the level info

        Args:
            msg: Message to print/ log
            logger: Logger which should log

        """
        self.loggers[logger].info(msg)

    def debug(self, msg, logger="default"):
        """
        Prints and logs a message with the level debug

        Args:
            msg: Message to print/ log
            logger: Logger which should log

        """
        self.loggers[logger].debug(msg)

    def error(self, msg, logger="default"):
        """
        Prints and logs a message with the level error

        Args:
            msg: Message to print/ log
            logger: Logger which should log

        """
        self.loggers[logger].error(msg)

    def log_to(self, msg, name, log_to_default=False):
        """
        Logs to an existing logger or if logger does not exists creates new one

        Args:
            msg: Message to be logged
            name: Name of the logger to log to (usually also the logfile-name)
            log_to_default: Flag if it should in addition to the logger given by name, log to the default logger

        """

        if name not in self.loggers:
            self.add_logger(name)
        self.log(msg, name)
        if log_to_default:
            self.log(msg, "default")

    def show_text(self, text, name=None, logger="default", **kwargs):
        """
        Logs a text. Calls the log function (for combatibility reasons with AbstractLogger)

        Args:
            text: Text to be logged
            name: Some identifier for the text (will be added infront of the text)
            logger: Name of the Logger to log to

        """
        if name is not None:
            self.log("{}: {}".format(name, text), logger)
        else:
            self.log(text, logger)

    def show_value(self, value, name=None, logger="default", **kwargs):
        """
        Logs a Value. Calls the log function (for combatibility reasons with AbstractLogger)

        Args:
            value: Value to be logged
            name: Some identifier for the text (will be added infront of the text)
            logger: Name of the Logger to log to

        """
        if name is not None:
            self.log("{}: {}".format(name, value), logger)
        else:
            self.log(value, logger)
