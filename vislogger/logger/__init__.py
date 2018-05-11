from vislogger.logger.abstractlogger import AbstractLogger
from vislogger.logger.combinedlogger import CombinedLogger
from vislogger.logger.plt.numpyseabornplotlogger import NumpySeabornPlotLogger
from vislogger.logger.file.numpyplotfilelogger import NumpyPlotFileLogger
from vislogger.logger.file.textlogger import TextLogger
from vislogger.logger.visdom.numpyvisdomlogger import NumpyVisdomLogger
from vislogger.logger.experiment.experimentlogger import ExperimentLogger

try:
    from vislogger.logger.experiment.pytorchexperimentlogger import PytorchExperimentLogger
    from vislogger.logger.file.pytorchplotfilelogger import PytorchPlotFileLogger
    from vislogger.logger.visdom.pytorchvisdomlogger import PytorchVisdomLogger
except ImportError as e:
    print("Could not import Pytorch related modules.")
    print(e)

try:
    from vislogger.logger.message.telegramlogger import TelegramLogger
except ImportError as e:
    print("Could not import Telegram related modules.")
    print(e)
