from trixi.experiment.experiment import Experiment

try:
    from trixi.experiment.pytorchexperiment import PytorchExperiment
except ImportError as e:
    import warnings
    warnings.warn(ImportWarning("Could not import Pytorch related modules:\n%s"
        % e.msg))
