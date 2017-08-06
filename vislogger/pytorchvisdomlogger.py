import tempfile

import numpy as np
import torch
from graphviz import Digraph
from torch.autograd import Variable

from vislogger import NumpyVisdomLogger


class PytorchVisdomLogger(NumpyVisdomLogger):
    """
    Visual logger, inherits the NumpyVisdomLogger and plots/ logs pytorch tensors and variables on a Visdom server.
    """

    def process_params(self, f, *args, **kwargs):
        """
        Inherited "decorator": convert Pytorch variables and Tensors to numpy arrays
        """

        ### convert args
        args = (a.cpu().numpy() if type(a).__module__ == 'torch' else a for a in args)
        args = (a.data.cpu().numpy() if isinstance(a, Variable) else a for a in args)

        ### convert kwargs
        for key, data in kwargs.items():
            if isinstance(data, Variable):
                kwargs[key] = data.data.cpu().numpy()
            elif type(data).__module__ == 'torch':
                kwargs[key] = data.cpu().numpy()

        return f(self, *args, **kwargs)

    def plot_model_statistics(self, model, env_app=None, model_name="", plot_grad=False):
        """
        Plots statstics (mean, std, abs(max)) of the weights or the corresponding gradients of a model as a barplot

        :param model: Model with the weights
        :param env_app: visdom environment name appendix, if none is given, it uses "-histogram"
        :param model_name: Name of the model (is used as window name)
        :param plot_grad: If false plots weight statistics, if true plot the gradients of the weights
        """

        if env_app is None:
            env_app = "-histogram"

        means = []
        stds = []
        maxmin = []
        legendary = []
        for i, (m_param_name, m_param) in enumerate(model.named_parameters()):
            win_name = "%s_params" % model_name
            if plot_grad:
                m_param = m_param.grad
                win_name = "%s_grad" % model_name

            param_mean = m_param.data.mean()
            param_std = np.sqrt(m_param.data.var())

            means.append(param_mean)
            stds.append(param_std)
            maxmin.append(torch.max(torch.abs(m_param)).data[0])
            legendary.append("%s-%s" % (model_name, m_param_name))

        self.show_barplot(name=win_name, array=np.asarray([means, stds, maxmin]), legend=legendary,
                          rownames=["mean", "std", "max"], env_app=env_app)

    def plot_model_statistics_weights(self, model, env_app=None, model_name=""):
        """
        Plots statstics (mean, std, abs(max)) of the weights of a model as a barplot (uses plot model statistics with
        plot_grad=False  )

        :param model: Model with the weights
        :param env_app: visdom environment name appendix, if none is given, it uses "-histogram"
        :param model_name: Name of the model (is used as window name)
        """
        self.plot_model_statistics(model=model, env_app=env_app, model_name=model_name, plot_grad=False)

    def plot_model_statistics_grads(self, model, env_app=None, model_name=""):
        """
        Plots statstics (mean, std, abs(max)) of the gradients of a model as a barplot (uses plot model statistics with
        plot_grad=True )

        :param model: Model with the weights and the corresponding gradients (have to calculated previously)
        :param env_app: visdom environment name appendix, if none is given, it uses "-histogram"
        :param model_name: Name of the model (is used as window name)
        """
        self.plot_model_statistics(model=model, env_app=env_app, model_name=model_name, plot_grad=True)

    def plot_mutliple_models_statistics_weights(self, model_dict, env_app=None):
        """
        Given models in a dict, plots the weight statistics of the models.

        :param model_dict: Dict with models, the key is assumed to be the name, while the value is the model
        :param env_app: visdom environment name appendix, if none is given, it uses "-histogram"
        """
        for model_name, model in model_dict.items():
            self.plot_model_statistics_weights(model=model, env_app=env_app, model_name=model_name)

    def plot_mutliple_models_statistics_grads(self, model_dict, env_app=None):
        """
        Given models in a dict, plots the gradient statistics of the models.

        :param model_dict: Dict with models, the key is assumed to be the name, while the value is the model
        :param env_app: visdom environment name appendix, if none is given, it uses "-histogram"
        """
        for model_name, model in model_dict.items():
            self.plot_model_statistics_grads(model=model, env_app=env_app, model_name=model_name)

    def plot_model_structure(self, model, input_size, name=None, use_cuda=True, delete_tmp_on_close=False):
        """
        Plots the model structure/ model graph of a pytorch module (this only works correctly with pytorch 0.2.0).

        :param model: The graph of this model will be plotted
        :param input_size: Input size of the model (with batch dim)
        :param name: The name of the window in the visdom env
        :param use_cuda: Perform model dimension calculations on the gpu (cuda)
        :param delete_tmp_on_close: Determines if the tmp file will be deleted on close. If set true, can cause problems
        due to the multi threadded plotting.
        """

        def make_dot(var, state_dict=None):
            """ Produces Graphviz representation of PyTorch autograd graph
            Blue nodes are the Variables that require grad, orange are Tensors
            saved for backward in torch.autograd.Function
            Args:
                var: output Variable
                state_dict: dict of (name, parameter) to add names to node that require grad
            """
            if state_dict is not None:
                # assert isinstance(params.values()[0], Variable)
                param_map = {id(v): k for k, v in state_dict.items()}

            node_attr = dict(style='filled',
                             shape='box',
                             align='left',
                             fontsize='12',
                             ranksep='0.1',
                             height='0.2')
            dot = Digraph(node_attr=node_attr, graph_attr=dict(size="12,12"), format="svg")
            seen = set()

            def size_to_str(size):
                return '(' + (', ').join(['%d' % v for v in size]) + ')'

            def add_nodes(var):
                if var not in seen:
                    if torch.is_tensor(var):
                        dot.node(str(id(var)), size_to_str(var.size()), fillcolor='orange')
                    elif hasattr(var, 'variable'):
                        u = var.variable
                        if state_dict is not None and id(u.data) in param_map:
                            name = param_map[id(u.data)]
                        else:
                            name = ""
                        node_name = '%s\n %s' % (name, size_to_str(u.size()))
                        dot.node(str(id(var)), node_name, fillcolor='lightblue')
                    else:
                        name = str(type(var).__name__)
                        if name.endswith("Backward"):
                            name = name[:-8]
                        dot.node(str(id(var)), name)
                    seen.add(var)
                    if hasattr(var, 'next_functions'):
                        for u in var.next_functions:
                            if u[0] is not None:
                                dot.edge(str(id(u[0])), str(id(var)))
                                add_nodes(u[0])
                    if hasattr(var, 'saved_tensors'):
                        for t in var.saved_tensors:
                            dot.edge(str(id(t)), str(id(var)))
                            add_nodes(t)

            add_nodes(var.grad_fn)
            return dot

        # Create input
        inputs = Variable(torch.randn(input_size))
        if use_cuda:
            inputs = inputs.cuda()
            model = model.cuda()

        # get output
        output = model(inputs)

        # get temp file to store svg in
        fp = tempfile.NamedTemporaryFile(suffix=".svg", delete=delete_tmp_on_close)

        # Create model graph and store it as svg
        g = make_dot(output, model.state_dict())
        x = g.render(fp.name[:-4], cleanup=True)

        # Display model graph in visdom
        self.show_svg(svg=x, name=name)

