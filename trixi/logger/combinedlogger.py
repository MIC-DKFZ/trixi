from collections import defaultdict

from trixi.logger.abstractlogger import AbstractLogger


def create_function(self, sub_methods):
    def surrogate_fn(*args, **kwargs):

        for sub_method in sub_methods:

            try:

                method_cntr = self.log_methods_cntr[sub_method]
                method_freq = self.log_methods_freq[sub_method]

                if method_freq is None or method_freq == 0:
                    continue

                use_name = False
                if "name" in kwargs and not ("ignore_name_in_args" in kwargs and kwargs["ignore_name_in_args"] is True):
                    name_id = kwargs["name"]
                    if "tag" in kwargs:
                        name_id = name_id + kwargs["tag"]
                    method_cntr = self.log_methods_name_cntr[sub_method][name_id]
                    use_name = True

                if "log_all" in kwargs and kwargs["log_all"] is True:
                    sub_method(*args, **kwargs)
                    continue

                if method_cntr % method_freq == 0:
                    sub_method(*args, **kwargs)

                elif "same_as_last" in kwargs and kwargs["same_as_last"] is True:
                    if method_cntr % method_freq == 1:
                        sub_method(*args, **kwargs)
                    kwargs["do_not_increase"] = True

                if use_name:
                    self.log_methods_name_cntr[sub_method][kwargs["name"]] += 1
                elif "do_not_increase" not in kwargs or ("do_not_increase" in kwargs and kwargs["do_not_increase"] is
                                                         False):
                    self.log_methods_cntr[sub_method] += 1

            except Exception as e:
                print("a combi logger method failed: ", str(sub_method))

    return surrogate_fn


class CombinedLogger(object):
    """
    A Logger which can combine all other logger and if called calls all the sub loggers

    """

    def __init__(self, *loggers):
        """
        Initializes a new combined logger with a list of logger and their logging frequencies.

        By default it initalized a counter for each logger method, and on each call it increases the counter (and if the
        counter is a multiple of the frequency it logs to the given logger).

        Furthermore you can use the same_as_last argument in all methods to repeat the frequency_counter from the
        previous call. If you set the do_not_increase attribute to True, it wont increase the counter, it has been
        called 10 times and the frequency is 10, it will log, and also log in the next round, since the counter is
        still 10 and was not increase. Use can use the log_all attribute, it you want to ignore the frequencies and
        simply log to all loggers. By default, if your method has a name attribute, there is a counter for each
        method called with a unique name (so one counter for plot(name="1") and a different counter for plot(name="2")),
        but if you want one counter for the method (despite having a name attribute), you can set the
        ignore_name_in_args to True,

        Args:
            *loggers (list): a list of tuples where each tuple conisist of (logger, frequencies), where logger is
            a given logger and frequencies is the frequency (or rather log each on in frequencies, e.g. if it is 10,
            it logs every ten_th call).
        """

        self.loggers, self.frequencies = zip(*loggers)

        for logger in self.loggers:
            if not isinstance(logger, AbstractLogger):
                raise TypeError("All logger must be subclasses of the abstract visual logger.")
        for freq in self.frequencies:
            if freq is None:
                continue
            elif freq < 0:
                raise ValueError("All frequencies must be positive.")

        self.logger_methods = defaultdict(list)
        self.log_methods_cntr = defaultdict(dict)
        self.log_methods_freq = defaultdict(int)
        self.log_methods_name_cntr = defaultdict(int)

        for logger, freq in zip(self.loggers, self.frequencies):
            if freq is None:
                continue

            logger_vars = [i for i in dir(logger) if not i.startswith("__")]

            for el in logger_vars:
                if hasattr(logger, el) and callable(getattr(logger, el)):
                    self.logger_methods[el].append(getattr(logger, el))
                    self.log_methods_cntr[getattr(logger, el)] = 0
                    self.log_methods_freq[getattr(logger, el)] = freq
                    self.log_methods_name_cntr[getattr(logger, el)] = defaultdict(int)

        for method_name, sub_methods in self.logger_methods.items():
            setattr(self, method_name, create_function(self, sub_methods))
