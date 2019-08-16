from trixi.logger.tensorboard.tensorboardxlogger import TensorboardXLogger
try:
    from trixi.logger.tensorboard.pytorchtensorboardxlogger import PytorchTensorboardXLogger
except ImportError as e:
    import warnings
    warnings.warn(ImportWarning("Could not import Pytorch related modules:\n%s"
        % e.msg))
