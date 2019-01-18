from trixi.logger.file.numpyplotfilelogger import NumpyPlotFileLogger
from trixi.logger.file.textfilelogger import TextFileLogger

try:
    from trixi.logger.file.pytorchplotfilelogger import PytorchPlotFileLogger
except ImportError as e:
    print("Could not import Pytorch related modules.")
    print(e)
