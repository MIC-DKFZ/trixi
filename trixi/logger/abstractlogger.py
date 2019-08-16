from abc import ABCMeta, abstractmethod
import _thread
from functools import wraps


def convert_params(f):
    """Decorator to call the process_params method of the class."""

    @wraps(f)
    def wrapper(self, *args, **kwargs):
        return self.process_params(f, *args, **kwargs)

    return wrapper


def threaded(f):
    """Decorator to run the process in an extra thread."""

    def wrapper(*args, **kwargs):
        return _thread.start_new(f, args, kwargs)

    return wrapper


class AbstractLogger(object):
    """Abstract interface for visual logger."""

    __metaclass__ = ABCMeta

    @abstractmethod
    def __init__(self, *args, **kwargs):
        pass

    def process_params(self, f, *args, **kwargs):
        """
        Implement this to handle data conversions in your logger.

        Example: Implement logger for numpy data, then implement torch logger as child of numpy logger and just use
        the process_params method to convert from torch to numpy.
        """

        return f(self, *args, **kwargs)

    @abstractmethod
    @convert_params
    def show_image(self, *args, **kwargs):
        """Abstract method which should handle and somehow log/ store an image"""
        raise NotImplementedError()

    @abstractmethod
    @convert_params
    def show_value(self, *args, **kwargs):
        """Abstract method which should handle and somehow log/ store a value"""
        raise NotImplementedError()

    @abstractmethod
    @convert_params
    def show_text(self, *args, **kwargs):
        """Abstract method which should handle and somehow log/ store a text"""
        raise NotImplementedError()

    @abstractmethod
    @convert_params
    def show_barplot(self, *args, **kwargs):
        """Abstract method which should handle and somehow log/ store a barplot"""
        raise NotImplementedError()

    @abstractmethod
    @convert_params
    def show_lineplot(self, *args, **kwargs):
        """Abstract method which should handle and somehow log/ store a lineplot"""
        raise NotImplementedError()

    @abstractmethod
    @convert_params
    def show_scatterplot(self, *args, **kwargs):
        """Abstract method which should handle and somehow log/ store a scatterplot"""
        raise NotImplementedError()

    @abstractmethod
    @convert_params
    def show_piechart(self, *args, **kwargs):
        """Abstract method which should handle and somehow log/ store a piechart"""
        raise NotImplementedError()
