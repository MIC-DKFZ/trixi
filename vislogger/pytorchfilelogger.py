import torch
from torch.autograd import Variable

from vislogger.numpyfilelogger import NumpyFileLogger


class PytorchFileLogger(NumpyFileLogger):
    """
    Visual logger, inherits the NumpyFileLogger and plots/ logs pytorch tensors and variables as files on the local
    file system.
    """


    def __init__(self, *args, **kwargs):
        super(PytorchFileLogger, self).__init__(*args, **kwargs)


    def process_params(self, f, *args, **kwargs):
        """
        Inherited "decorator": convert Pytorch variables and Tensors to numpy arrays
        """

        ### convert args
        args = (a.cpu().numpy() if torch.is_tensor(a) else a for a in args)
        args = (a.data.cpu().numpy() if isinstance(a, Variable) else a for a in args)

        ### convert kwargs
        for key, data in kwargs.items():
            if isinstance(data, Variable):
                kwargs[key] = data.data.cpu().numpy()
            elif torch.is_tensor(data):
                kwargs[key] = data.cpu().numpy()

        return f(self, *args, **kwargs)