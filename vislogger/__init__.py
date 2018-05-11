use_agg = True
import matplotlib
if use_agg: matplotlib.use("Agg")

from vislogger.config import Config
from vislogger.abstractlogger import AbstractLogger
from vislogger.combinedlogger import CombinedLogger
from vislogger.textlogger import TextLogger
from vislogger.numpyplotfilelogger import NumpyPlotFileLogger
from vislogger.numpyseabornplotlogger import NumpySeabornPlotLogger
from vislogger.experimentlogger import ExperimentLogger
from vislogger.experiment import Experiment
from vislogger.telegramlogger import TelegramLogger

# pynvml
try:
    from vislogger.gpu_monitor import GpuMonitor
except ImportError as e:
    print("Could not import pynvml related modules.")
    print(e)

# Visdom
try:
    from vislogger.extravisdom import ExtraVisdom
    from vislogger.numpyvisdomlogger import NumpyVisdomLogger
except ImportError as e:
    print("Could not import Visdom related modules.")
    print(e)

# Pytorch
try:
    from vislogger.pytorchplotfilelogger import PytorchPlotFileLogger
    from vislogger.pytorchvisdomlogger import PytorchVisdomLogger
    from vislogger.pytorchexperimentlogger import PytorchExperimentLogger
    from vislogger.experiment import PytorchExperiment
except ImportError as e:
    print("Could not import Pytorch related modules.")
    print(e)

# Telegram
try:
    from vislogger.telegramlogger import TelegramLogger
except ImportError as e:
    print("Could not import Telegram related modules.")
    print(e)
