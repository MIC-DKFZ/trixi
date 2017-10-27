import string
import random
import os


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


def random_string(length):
    return "".join(random.choice(string.ascii_letters + string.digits) for _ in range(length))


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


def name_and_iter_to_filename(name, n_iter, ending, iter_format="{:05d}", prefix=False):

    iter_str = iter_format.format(n_iter)
    if prefix:
        name = iter_str + "_" + name + ending
    else:
        name = name + "_" + iter_str + ending

    return name