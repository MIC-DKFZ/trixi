from vislogger.logger.abstractlogger import AbstractLogger
from vislogger.logger.combinedlogger import CombinedLogger
from vislogger.logger.experiment import ExperimentLogger
from vislogger.logger.file import NumpyPlotFileLogger, TextLogger
from vislogger.logger.plt import NumpySeabornPlotLogger
from vislogger.logger.visdom import NumpyVisdomLogger

try:
    from vislogger.logger.experiment import PytorchExperimentLogger
    from vislogger.logger.file import PytorchPlotFileLogger
    from vislogger.logger.visdom import PytorchVisdomLogger
except ImportError as e:
    print("Could not import Pytorch related modules.")
    print(e)

try:
    from vislogger.logger.message import TelegramLogger
except ImportError as e:
    print("Could not import Telegram related modules.")
    print(e)
