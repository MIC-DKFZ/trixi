from trixi.logger.tensorboard.tensorboardlogger import TensorboardLogger
try:
    from trixi.logger.tensorboard.pytorchtensorboardlogger import PytorchTensorboardLogger
except ImportError as e:
    import warnings
    warnings.warn(ImportWarning("Could not import Pytorch related modules:\n%s"
        % e.msg))
