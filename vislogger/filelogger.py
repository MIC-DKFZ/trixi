import datetime
import logging
import os

from vislogger.abstractvisuallogger import AbstractVisualLogger


def create_folder(path):
    """
    Creates a folder if not already exits
    Args:
        :param path: The folder to be created
    Returns
        :return: True if folder was newly created, false if folder already exists
    """

    if not os.path.exists(path):
        os.makedirs(path)
        return True
    else:
        return False


class Singleton:
    """
    A non-thread-safe helper class to ease implementing singletons.
    This should be used as a decorator -- not a metaclass -- to the
    class that should be a singleton.

    The decorated class can define one `__init__` function that
    takes only the `self` argument. Also, the decorated class cannot be
    inherited from. Other than that, there are no restrictions that apply
    to the decorated class.

    To get the singleton instance, use the `Instance` method. Trying
    to use `__call__` will result in a `TypeError` being raised.

    """

    _instance = None

    def __init__(self, decorated):
        self._decorated = decorated

    def get_instance(self, **kwargs):
        """
        Returns the singleton instance. Upon its first call, it creates a
        new instance of the decorated class and calls its `__init__` method.
        On all subsequent calls, the already created instance is returned.

        """
        if not self._instance:
            self._instance = self._decorated(**kwargs)
            return self._instance
        else:
            return self._instance

    def __call__(self):
        raise TypeError('Singletons must be accessed through `get_instance()`.')
        # return self.get_instance()

    def __instancecheck__(self, inst):
        return isinstance(inst, self._decorated)


#@Singleton
class FileLogger(AbstractVisualLogger):
    """A single class for logging"""

    def __init__(self, path=None, **kwargs):
        
        super(FileLogger, self).__init__( **kwargs)

        self.logger = logging.getLogger("output")  #
        self.logger.setLevel(logging.DEBUG)

        logger_handler = logging.StreamHandler()  # Handler for the logger
        self.logger.addHandler(logger_handler)

        # First, generic formatter:
        self.clean_formatter = logging.Formatter('%(message)s')
        logger_handler.setFormatter(self.clean_formatter)

        self.file_handler = None
        self.file_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

        self.aux_loggers = {"output": self.logger}

        self.has_folder = False
        if path is not None:
            self.make_log_folder(path)
            self.set_output_file(os.path.join(self.logfile_dir, "output.log"))
            self.has_folder = True

    def make_log_folder(self, path):
        """Creates a new log folder"""

        run = 0
        while os.path.exists(os.path.join(path, "run-%05d" % run)):
            run += 1

        self.run = run
        self.base_dir = os.path.join(path, "run-%05d" % run)
        self.logfile_dir = os.path.join(self.base_dir, "logs")
        self.model_dir = os.path.join(self.base_dir, "models")
        self.image_dir = os.path.join(self.base_dir, "imgs")
        self.plot_dir = os.path.join(self.base_dir, "plots")
        self.store_dir = os.path.join(self.base_dir, "store")

        self.store_dirs = dict()

        create_folder(self.base_dir)
        create_folder(self.logfile_dir)
        create_folder(self.model_dir)
        create_folder(self.image_dir)
        create_folder(self.plot_dir)
        create_folder(self.store_dir)

        with open(os.path.join(self.base_dir, "time.txt"), 'w') as output:
            now = datetime.datetime.now()
            output.write(now.strftime("%y-%m-%d_%H:%M:%S"))

    def set_output_file(self, file):
        """Sets a new output file"""
        assert file is not None

        if self.file_handler is not None:
            self.logger.removeHandler(self.file_handler)

        self.file_handler = logging.FileHandler(file)
        self.file_handler.setLevel(logging.DEBUG)
        self.file_handler.setFormatter(self.file_formatter)
        self.logger.addHandler(self.file_handler)

    def add_log_file(self, name, formatter="clean"):
        """Adds a new auxilary log file with the name 'name.log' """
        assert name is not None and isinstance(name, str)

        if not self.has_folder:
            return False

        if name in self.aux_loggers:
            self.error("Log file with this name already exists !")
            return False

        if formatter == "clean":
            formatter = self.clean_formatter
        elif formatter == "file":
            formatter = self.file_formatter

        file = os.path.join(self.logfile_dir, name + ".log")

        aux_logger = logging.getLogger(name)
        aux_logger.setLevel(logging.DEBUG)

        aux_file_handler = logging.FileHandler(file)
        aux_file_handler.setFormatter(formatter)
        aux_logger.addHandler(aux_file_handler)

        self.aux_loggers[name] = aux_logger

        return True

    def get_storage_folder(self, name, *paths):
        """Makes and returns a new folder in the base_dir for the given folder path"""
        assert name is not None

        if not self.has_folder:
            return False

        if name not in self.store_dirs:
            if len(paths) == 0:
                paths = name
            target_dir = os.path.join(self.store_dir_dir, paths)
            self.store_dirs[name] = target_dir
            create_folder(target_dir)
        else:
            target_dir = self.store_dirs[name]

        return target_dir

    def print(self, *args):
        """Prints and logs an object"""
        for text in args:
            self.log(str(text))

    def log(self, msg):
        """Logs a string as info log"""
        self.logger.info(msg)

    def info(self, msg):
        """wrapper for logger.info"""
        self.logger.info(msg)

    def debug(self, msg):
        """wrapper for logger.debug"""
        self.logger.debug(msg)

    def error(self, msg):
        """wrapper for logger.error"""
        self.logger.error(msg)

    def log_to(self, name, msg, log_level=logging.INFO, log_to_output=True):
        """Logs to an previously created file"""

        if name not in self.aux_loggers:
            self.error("Create/Add a log file first with 'add_log_files(name)' !")
            return

        aux_logger = self.aux_loggers[name]
        aux_logger.log(log_level, msg)
        if log_to_output:
            self.logger.log(log_level, msg)


    def show_text(self, text, *args, **kwargs):
        self.log(text)

    def show_value(self, value, *args, **kwargs):
        self.log(value)