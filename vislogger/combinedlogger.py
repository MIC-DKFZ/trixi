from collections import defaultdict


def create_function(self, sub_methods):

    def surrogate_fn(*args, **kwargs):

        for sub_method in sub_methods:

                method_cntr = self.log_methods_cntr[sub_method]
                method_freq = self.log_methods_freq[sub_method]

                if method_cntr % method_freq == 0:
                    sub_method(*args, **kwargs)

                self.log_methods_cntr[sub_method] += 1

    return surrogate_fn


class CombinedLogger():

    def __init__(self, *loggers):

        self.loggers, self.frequency = zip(*loggers)

        self.logger_methods = defaultdict(list)
        self.log_methods_cntr = defaultdict(int)
        self.log_methods_freq = defaultdict(int)


        for logger, freq in zip(self.loggers, self.frequency):

            logger_vars = [i for i in dir(logger) if not i.startswith("__")]

            for el in logger_vars:
                if hasattr(logger, el) and callable(getattr(logger, el)):
                    self.logger_methods[el].append(getattr(logger, el))
                    self.log_methods_cntr[getattr(logger, el)] = 0
                    self.log_methods_freq[getattr(logger, el)] = freq

        for method_name, sub_methods in self.logger_methods.items():

            setattr(self, method_name, create_function(self, sub_methods))

        print("xD")






