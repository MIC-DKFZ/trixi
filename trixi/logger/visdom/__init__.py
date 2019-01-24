from trixi.logger.visdom.numpyvisdomlogger import NumpyVisdomLogger
try:
    from trixi.logger.visdom.pytorchvisdomlogger import PytorchVisdomLogger
except ImportError as e:
    print("Could not import Pytorch related modules.")
    print(e)
