from trixi.logger.file.numpyplotfilelogger import NumpyPlotFileLogger
from trixi.logger.file.textfilelogger import TextFileLogger

try:
    from trixi.logger.file.pytorchplotfilelogger import PytorchPlotFileLogger
except ImportError as e:
    import warnings
    warnings.warn(ImportWarning("Could not import Pytorch related modules:\n%s"
        % e.msg))
