from trixi.logger.experiment.experimentlogger import ExperimentLogger
try:
    from trixi.logger.experiment.pytorchexperimentlogger import PytorchExperimentLogger
except ImportError as e:
    import warnings
    warnings.warn(ImportWarning("Could not import Pytorch related modules:\n%s"
        % e.msg))
