import tempfile
from multiprocessing import Process

import numpy as np
import torch
from graphviz import Digraph
from torch.autograd import Variable
from torchvision.utils import make_grid

from vislogger import NumpyVisdomLogger
from vislogger.abstractlogger import convert_params


class PytorchVisdomLogger(NumpyVisdomLogger):
    """
    Visual logger, inherits the NumpyVisdomLogger and plots/ logs pytorch tensors and variables on a Visdom server.
    """

    def __init__(self, *args, **kwargs):
        super(PytorchVisdomLogger, self).__init__(*args, **kwargs)

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

    def plot_model_statistics(self, model, env_appendix=None, model_name="", plot_grad=False, **kwargs):
        """
        Plots statstics (mean, std, abs(max)) of the weights or the corresponding gradients of a model as a barplot

        :param model: Model with the weights
        :param env_appendix: visdom environment name appendix, if none is given, it uses "-histogram"
        :param model_name: Name of the model (is used as window name)
        :param plot_grad: If false plots weight statistics, if true plot the gradients of the weights
        """

        if env_appendix is None:
            env_appendix = "-histogram"

        means = []
        stds = []
        maxmin = []
        legendary = []
        for i, (m_param_name, m_param) in enumerate(model.named_parameters()):
            win_name = "%s_params" % model_name
            if plot_grad:
                m_param = m_param.grad
                win_name = "%s_grad" % model_name

            if m_param is not None:
                param_mean = m_param.data.mean()
                param_std = np.sqrt(m_param.data.var())

                means.append(param_mean)
                stds.append(param_std)
                maxmin.append(torch.max(torch.abs(m_param)).data[0])
                legendary.append("%s-%s" % (model_name, m_param_name))

        self.show_barplot(name=win_name, array=np.asarray([means, stds, maxmin]), legend=legendary,
                          rownames=["mean", "std", "max"], env_appendix=env_appendix)

    def plot_model_statistics_weights(self, model, env_appendix=None, model_name="", **kwargs):
        """
        Plots statstics (mean, std, abs(max)) of the weights of a model as a barplot (uses plot model statistics with
        plot_grad=False  )

        :param model: Model with the weights
        :param env_appendix: visdom environment name appendix, if none is given, it uses "-histogram"
        :param model_name: Name of the model (is used as window name)
        """
        self.plot_model_statistics(model=model, env_appendix=env_appendix, model_name=model_name, plot_grad=False)

    def plot_model_statistics_grads(self, model, env_appendix=None, model_name="", **kwargs):
        """
        Plots statstics (mean, std, abs(max)) of the gradients of a model as a barplot (uses plot model statistics with
        plot_grad=True )

        :param model: Model with the weights and the corresponding gradients (have to calculated previously)
        :param env_appendix: visdom environment name appendix, if none is given, it uses "-histogram"
        :param model_name: Name of the model (is used as window name)
        """
        self.plot_model_statistics(model=model, env_appendix=env_appendix, model_name=model_name, plot_grad=True)

    def plot_mutliple_models_statistics_weights(self, model_dict, env_appendix=None, **kwargs):
        """
        Given models in a dict, plots the weight statistics of the models.

        :param model_dict: Dict with models, the key is assumed to be the name, while the value is the model
        :param env_appendix: visdom environment name appendix, if none is given, it uses "-histogram"
        """
        for model_name, model in model_dict.items():
            self.plot_model_statistics_weights(model=model, env_appendix=env_appendix, model_name=model_name)

    def plot_mutliple_models_statistics_grads(self, model_dict, env_appendix=None, **kwargs):
        """
        Given models in a dict, plots the gradient statistics of the models.

        :param model_dict: Dict with models, the key is assumed to be the name, while the value is the model
        :param env_appendix: visdom environment name appendix, if none is given, it uses "-histogram"
        """
        for model_name, model in model_dict.items():
            self.plot_model_statistics_grads(model=model, env_appendix=env_appendix, model_name=model_name)

    def plot_model_structure(self, model, input_size, name=None, use_cuda=True, delete_tmp_on_close=False, **kwargs):
        """
        Plots the model structure/ model graph of a pytorch module (this only works correctly with pytorch 0.2.0).

        :param model: The graph of this model will be plotted
        :param input_size: Input size of the model (with batch dim)
        :param name: The name of the window in the visdom env
        :param use_cuda: Perform model dimension calculations on the gpu (cuda)
        :param delete_tmp_on_close: Determines if the tmp file will be deleted on close. If set true, can cause problems
        due to the multi threadded plotting.
        """

        def make_dot(output_var, state_dict=None):
            """ Produces Graphviz representation of PyTorch autograd graph
            Blue nodes are the Variables that require grad, orange are Tensors
            saved for backward in torch.autograd.Function
            Args:
                output_var: output Variable
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
                            node_name = param_map[id(u.data)]
                        else:
                            node_name = ""
                        node_name = '%s\n %s' % (node_name, size_to_str(u.size()))
                        dot.node(str(id(var)), node_name, fillcolor='lightblue')
                    else:
                        node_name = str(type(var).__name__)
                        if node_name.endswith("Backward"):
                            node_name = node_name[:-8]
                        dot.node(str(id(var)), node_name)
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

            add_nodes(output_var.grad_fn)
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

    def show_image_grid(self, images, name=None, title=None, caption=None, env_appendix="", opts=None,
                        image_args=None, **kwargs):

        if opts is None: opts = {}
        if image_args is None: image_args = {}

        tensor = images.cpu()
        viz_task = {
            "type": "image_grid",
            "tensor": tensor,
            "name": name,
            "title": title,
            "caption": caption,
            "env_appendix": env_appendix,
            "opts": opts,
            "image_args": image_args
        }
        self._queue.put_nowait(viz_task)

    def __show_image_grid(self, tensor, name=None, title=None, caption=None, env_appendix="", opts=None,
                          image_args=None, **kwargs):

        if opts is None: opts = {}
        if image_args is None: image_args = {}

        if isinstance(tensor, Variable):
            tensor = tensor.data

        assert torch.is_tensor(tensor), "tensor has to be a pytorch tensor or variable"
        assert tensor.dim() == 4, "tensor has to have 4 dimensions"
        assert tensor.size(1) == 1 or tensor.size(
            1) == 3, "The 1. dimension (channel) has to be either 1 (gray) or 3 (rgb)"

        grid = make_grid(tensor, **image_args)
        image = grid.mul(255).clamp(0, 255).byte().numpy()

        opts = opts.copy()
        opts.update(dict(
            title=name,
            caption=caption
        ))

        win = self.vis.image(
            img=image,
            win=name,
            env=self.name + env_appendix,
            opts=opts
        )

        return win

    NumpyVisdomLogger.show_funcs["image_grid"] = __show_image_grid

    @convert_params
    def show_embedding(self, tensor, labels=None, name=None, method="tsne", n_dims=2, n_neigh=30, **meth_args):

        from sklearn import manifold
        import umap

        def __show_embedding(queue, tensor, labels=None, name=None, method="tsne", n_dims=2, n_neigh=30, **meth_args):
            emb_data = []

            linears = ['standard', 'ltsa', 'hessian', 'modified']
            if method in linears:

                loclin = manifold.LocallyLinearEmbedding(n_neigh, n_dims, method=method, **meth_args)
                emb_data = loclin.fit_transform(tensor)

            elif method == "isomap":
                iso = manifold.Isomap(n_neigh, n_dims, **meth_args)
                emb_data = iso.fit_transform(tensor)

            elif method == "mds":
                mds = manifold.MDS(n_dims, **meth_args)
                emb_data = mds.fit_transform(tensor)

            elif method == "spectral":
                se = manifold.SpectralEmbedding(n_components=n_dims, n_neighbors=n_neigh, **meth_args)
                emb_data = se.fit_transform(tensor)

            elif method == "tsne":
                tsne = manifold.TSNE(n_components=n_dims, perplexity=n_neigh, **meth_args)
                emb_data = tsne.fit_transform(tensor)

            elif method == "umap":
                um = umap.UMAP(n_components=n_dims, n_neighbors=n_neigh, **meth_args)
                emb_data = um.fit_transform(tensor)

            vis_task = {
                "type": "scatterplot",
                "array": emb_data,
                "labels": labels,
                "name": name,
                "env_appendix": "",
                "opts": {}
            }
            queue.put_nowait(vis_task)

        p = Process(target=__show_embedding, kwargs=dict(queue=self._queue,
                                                         tensor=tensor,
                                                         labels=labels,
                                                         name=name,
                                                         method=method,
                                                         n_dims=n_dims,
                                                         n_neigh=n_neigh,
                                                         **meth_args
                                                         ))
        p.start()


    @convert_params
    def show_roc_curve(self, tensor, labels, name):

        from sklearn import metrics

        def __show_roc_curve(self, tensor, labels, name):

            fpr, tpr, thresholds = metrics.roc_curve(labels.flatten(), tensor.flatten())
            self.show_lineplot(tpr, fpr, name=name, opts={"fillarea": True})
            self.add_to_graph(x_vals=np.arange(0, 1.1, 0.1), y_vals=np.arange(0, 1.1, 0.1), name=name, append=True)


        p = Process(target=__show_roc_curve, kwargs=dict(self=self,
                                                         tensor=tensor,
                                                         labels=labels,
                                                         name=name
                                                         ))
        p.start()
