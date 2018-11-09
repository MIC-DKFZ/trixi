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
    print("Could not import Pytorch related modules.")
    print(e)

try:
    from trixi.logger.message.telegrammessagelogger import TelegramMessageLogger
except ImportError as e:
    print("Could not import Telegram related modules.")
    print(e)
