from vislogger.abstractvisuallogger import AbstractVisualLogger
from vislogger.combinedlogger import CombinedLogger
from vislogger.extravisdom import ExtraVisdom
from vislogger.filelogger import FileLogger
from vislogger.numpyfilelogger import NumpyFileLogger
from vislogger.numpyseabornlogger import NumpySeabornLogger
from vislogger.numpyvisdomlogger import NumpyVisdomLogger

import imp
try:
    imp.find_module("torch")
    from vislogger.pytorchfilelogger import PytorchFileLogger
    from vislogger.pytorchvisdomlogger import PytorchVisdomLogger
except ImportError:
    pass

    
