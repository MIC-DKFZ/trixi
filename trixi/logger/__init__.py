from trixi.logger.abstractlogger import AbstractLogger
from trixi.logger.combinedlogger import CombinedLogger
from trixi.logger.plt.numpyseabornplotlogger import NumpySeabornPlotLogger
from trixi.logger.file.numpyplotfilelogger import NumpyPlotFileLogger
from trixi.logger.file.textfilelogger import TextFileLogger
from trixi.logger.visdom.numpyvisdomlogger import NumpyVisdomLogger
from trixi.logger.experiment.experimentlogger import ExperimentLogger

try:
    from trixi.logger.experiment.pytorchexperimentlogger import PytorchExperimentLogger
    from trixi.logger.file.pytorchplotfilelogger import PytorchPlotFileLogger
    from trixi.logger.visdom.pytorchvisdomlogger import PytorchVisdomLogger
except ImportError as e:
    import warnings
    warnings.warn(ImportWarning("Could not import Pytorch related modules:\n%s"
        % e.msg))

try:
    from trixi.logger.message.telegrammessagelogger import TelegramMessageLogger
except ImportError as e:
    import warnings
    warnings.warn(ImportWarning("Could not import Telegram related modules:\n%s"
        % e.msg))
