from trixi.experiment.experiment import Experiment

try:
    from trixi.experiment.pytorchexperiment import PytorchExperiment
except ImportError as e:
    print("Could not import Pytorch related modules.")
    print(e)
