from trixi.logger.experiment.experimentlogger import ExperimentLogger
try:
    from trixi.logger.experiment.pytorchexperimentlogger import PytorchExperimentLogger
except ImportError as e:
    print("Could not import Pytorch related modules.")
    print(e)
