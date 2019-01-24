from trixi.logger.visdom.numpyvisdomlogger import NumpyVisdomLogger
try:
    from trixi.logger.visdom.pytorchvisdomlogger import PytorchVisdomLogger
except ImportError as e:
    import warnings
    warnings.warn(ImportWarning("Could not import Pytorch related modules:\n%s"
        % e.msg))
