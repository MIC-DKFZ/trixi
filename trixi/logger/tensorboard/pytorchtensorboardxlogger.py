import torch
from trixi.logger.tensorboard.tensorboardxlogger import TensorboardXLogger


class PytorchTensorboardXLogger(TensorboardXLogger):
    """Abstract interface for visual logger."""

    def process_params(self, f, *args, **kwargs):
        """
        Inherited "decorator": convert PyTorch variables and Tensors to numpy arrays
        """

        # convert args
        args = (a.detach().cpu().numpy() if torch.is_tensor(a) else a for a in args)

        # convert kwargs
        for key, data in kwargs.items():
            if torch.is_tensor(data):
                kwargs[key] = data.detach().cpu().numpy()

        return f(self, *args, **kwargs)

    def __init__(self, *args, **kwargs):
        super(PytorchTensorboardXLogger, self).__init__(*args, **kwargs)

    def plot_model_structure(self, model, input_size):
        """
         Plots the model structure/ model graph of a pytorch module.

         Args:
             model: The graph of this model will be plotted.
             input_size: Input size of the model (with batch dim).
         """
        inpt_vars = [torch.randn(i_s) for i_s in input_size]

        self.writer.add_graph(model=model, input_to_model=inpt_vars)
