from vislogger.experiment.experiment import Experiment

try:
    from vislogger.experiment.pytorchexperiment import PytorchExperiment
except ImportError as e:
    print("Could not import Pytorch related modules.")
    print(e)
