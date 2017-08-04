from abc import ABCMeta, abstractmethod


def convert_params(f):
    """Decorator to call the process_params method of the class."""

    def wrapper(self, *args, **kwargs):
        return self.process_params(f, *args, **kwargs)

    return wrapper

class AbstractVisualLogger(object):
    __metaclass__ = ABCMeta
    """Abstract interface for visual logger."""

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
        raise NotImplementedError()

    @abstractmethod
    @convert_params
    def show_value(self, *args, **kwargs):
        raise NotImplementedError()

    @abstractmethod
    @convert_params
    def show_text(self, *args, **kwargs):
        raise NotImplementedError()

    @abstractmethod
    @convert_params
    def show_barplot(self, *args, **kwargs):
        raise NotImplementedError()

    @abstractmethod
    @convert_params
    def show_lineplot(self, *args, **kwargs):
        raise NotImplementedError()

    @abstractmethod
    @convert_params
    def show_scatterplot(self, *args, **kwargs):
        raise NotImplementedError()

    @abstractmethod
    @convert_params
    def show_piechart(self, *args, **kwargs):
        raise NotImplementedError()
