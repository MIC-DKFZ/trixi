import atexit
import tempfile
import warnings
from multiprocessing import Process

import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
import torch

from torch.autograd import Variable
from torchvision.utils import make_grid

from trixi.util.util import np_make_grid, get_tensor_embedding
from trixi.logger.experiment.pytorchexperimentlogger import PytorchExperimentLogger
from trixi.logger.visdom.numpyvisdomlogger import NumpyVisdomLogger, add_to_queue
from trixi.logger.abstractlogger import convert_params
from trixi.util.pytorchutils import get_guided_image_gradient, get_smooth_image_gradient, get_vanilla_image_gradient


from functools import wraps


def move_to_cpu(fn):
    """Decorator to call the process_params method of the class."""

    def __process_params(*args, **kwargs):
        ### convert args
        args = (a.detach().cpu() if torch.is_tensor(a) else a for a in args)
        ### convert kwargs
        for key, data in kwargs.items():
            if torch.is_tensor(data):
                kwargs[key] = data.detach().cpu()
        return fn(*args, **kwargs)

    # # @wraps(f)
    # def wrapper(*args, **kwargs):
    #     return __process_params(fn, *args, **kwargs)

    return __process_params

class PytorchVisdomLogger(NumpyVisdomLogger):
    """
    Visual logger, inherits the NumpyVisdomLogger and plots/ logs pytorch tensors and variables on a Visdom server.
    """

    def __init__(self, *args, **kwargs):
        super(PytorchVisdomLogger, self).__init__(*args, **kwargs)

    def process_params(self, f, *args, **kwargs):
        """
        Inherited "decorator": convert Pytorch variables and Tensors to numpy arrays.
        """

        ### convert args
        args = (a.detach().cpu().numpy() if torch.is_tensor(a) else a for a in args)
        # args = (a.data.cpu().numpy() if isinstance(a, Variable) else a for a in args)

        ### convert kwargs
        for key, data in kwargs.items():
            # if isinstance(data, Variable):
            #     kwargs[key] = data.detach().cpu().numpy()
            if torch.is_tensor(data):
                kwargs[key] = data.detach().cpu().numpy()

        return f(self, *args, **kwargs)


    @move_to_cpu
    def plot_model_statistics(self, model, env_appendix="", model_name="", plot_grad=False, **kwargs):
        """
        Plots statstics (mean, std, abs(max)) of the weights or the corresponding gradients of a model as a barplot.

        Args:
            model: Model with the weights.
            env_appendix: Visdom environment name appendix
            model_name: Name of the model (is used as window name).
            plot_grad: If false plots weight statistics, if true plot the gradients of the weights.
        """

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
                param_mean = m_param.detach().mean().item()
                param_std = m_param.detach().std().item()

                if np.isnan(param_std):
                    param_std = 0

                means.append(param_mean)
                stds.append(param_std)
                maxmin.append(torch.max(torch.abs(m_param)).item())
                legendary.append("%s-%s" % (model_name, m_param_name))

        self.show_barplot(name=win_name, array=np.asarray([means, stds, maxmin]), legend=legendary,
                          rownames=["mean", "std", "max"], env_appendix=env_appendix)

    def plot_model_statistics_weights(self, model, env_appendix="", model_name="", **kwargs):
        """
        Plots statstics (mean, std, abs(max)) of the weights of a model as a barplot (uses plot model statistics with plot_grad=False).

        Args:
            model: Model with the weights.
            env_appendix: Visdom environment name appendix
            model_name: Name of the model (is used as window name).
        """
        self.plot_model_statistics(model=model, env_appendix=env_appendix, model_name=model_name, plot_grad=False)

    def plot_model_statistics_grads(self, model, env_appendix="", model_name="", **kwargs):
        """
        Plots statstics (mean, std, abs(max)) of the gradients of a model as a barplot (uses plot model statistics with plot_grad=True).

        Args:
            model: Model with the weights and the corresponding gradients (have to calculated previously).
            env_appendix: Visdom environment name appendix
            model_name: Name of the model (is used as window name).
        """
        self.plot_model_statistics(model=model, env_appendix=env_appendix, model_name=model_name, plot_grad=True)

    def plot_model_gradient_flow(self, model, name="model", title=None):
        """
        Plots statstics (mean, std, abs(max)) of the weights or the corresponding gradients of a model as a barplot.

        Args:
            model: Model with the weights.
            env_appendix: Visdom environment name appendix, if none is given, it uses "-histogram".
            model_name: Name of the model (is used as window name).
            plot_grad: If false plots weight statistics, if true plot the gradients of the weights.
        """
        ave_grads = []
        layers = []

        named_parameters = model.named_parameters()
        for n, p in named_parameters:
            if (p.requires_grad) and ("bias" not in n):
                layers.append(n)
                ave_grads.append(p.grad.abs().mean())

        plt.figure()
        plt.plot(ave_grads, alpha=0.3, color="b")
        plt.hlines(0, 0, len(ave_grads) + 1, linewidth=1, color="k")
        plt.xticks(range(0, len(ave_grads), 1), layers, rotation="vertical")
        plt.xlim(xmin=0, xmax=len(ave_grads))
        plt.xlabel("Layers")
        plt.ylabel("average gradient {}".format(name))
        plt.title("Gradient flow")
        plt.grid(True)

        self.show_matplot_plt(plt.gcf(), name=name, title=title)

    def plot_mutliple_models_statistics_weights(self, model_dict, env_appendix=None, **kwargs):
        """
        Given models in a dict, plots the weight statistics of the models.

        Args:
            model_dict: Dict with models, the key is assumed to be the name, while the value is the model.
            env_appendix: Visdom environment name appendix
        """
        for model_name, model in model_dict.items():
            self.plot_model_statistics_weights(model=model, env_appendix=env_appendix, model_name=model_name)

    def plot_mutliple_models_statistics_grads(self, model_dict, env_appendix="", **kwargs):
        """
        Given models in a dict, plots the gradient statistics of the models.

        Args:
            model_dict: Dict with models, the key is assumed to be the name, while the value is the model.
            env_appendix: Visdom environment name appendix
        """
        for model_name, model in model_dict.items():
            self.plot_model_statistics_grads(model=model, env_appendix=env_appendix, model_name=model_name)

    def plot_model_structure(self, model, input_size, name="model_structure", use_cuda=True, delete_tmp_on_close=False, forward_kwargs=None, **kwargs):
        """
        Plots the model structure/ model graph of a pytorch module (this only works correctly with pytorch 0.2.0).

        Args:
            model: The graph of this model will be plotted.
            input_size: Input size of the model (with batch dim).
            name: The name of the window in the visdom env.
            use_cuda: Perform model dimension calculations on the gpu (cuda).
            delete_tmp_on_close: Determines if the tmp file will be deleted on close. If set true, can cause problems due to the multi threadded plotting.
        """

        if not hasattr(input_size[0], "__iter__"):
            input_size = [input_size, ]

        if not torch.cuda.is_available():
            use_cuda = False

        if forward_kwargs is None:
            forward_kwargs = {}

        def make_dot(output_var, state_dict=None):
            """
            Produces Graphviz representation of Pytorch autograd graph.
            Blue nodes are the Variables that require grad, orange are Tensors
            saved for backward in torch.autograd.Function.

            Args:
                output_var: output Variable
                state_dict: dict of (name, parameter) to add names to node that require grad
            """
            from graphviz import Digraph

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
        inpt_vars = [torch.randn(i_s) for i_s in input_size]
        if use_cuda:
            if next(model.parameters()).is_cuda:
                device = next(model.parameters()).device.index
            else:
                device = None
            inpt_vars = [i_v.cuda(device) for i_v in inpt_vars]
            model = model.cuda(device)

        # get output
        output = model(*inpt_vars, **forward_kwargs)

        # get temp file to store svg in
        fp = tempfile.NamedTemporaryFile(suffix=".svg", delete=delete_tmp_on_close)
        g = make_dot(output, model.state_dict())

        try:

            # Create model graph and store it as svg
            x = g.render(fp.name[:-4], cleanup=True)

            # Display model graph in visdom
            self.show_svg(svg=x, name=name)
        except Exception as e:
            warnings.warn("Could not render model, make sure the Graphviz executables are on your system.")

    @move_to_cpu
    @add_to_queue
    def show_image_grid(self, tensor, name=None, caption=None, env_appendix="", opts=None,
                          image_args=None, **kwargs):
        """
        Calls the save image grid method (for abstract logger combatibility)

        Args:
           images: 4d- tensor (N, C, H, W)
           name: The name of the window
           caption: Caption of the generated image grid
           env_appendix: appendix to the environment name, if used the new env is env+env_appendix
           opts: opts dict for the ploty/ visdom plot, i.e. can set window size, en/disable ticks,...
           image_args: Arguments for the tensorvision save image method


        """

        if opts is None: opts = {}
        if image_args is None: image_args = {}

        if isinstance(tensor, Variable):

            tensor = tensor.detach()

        if torch.is_tensor(tensor):
            assert torch.is_tensor(tensor), "tensor has to be a pytorch tensor or variable"
            assert tensor.dim() == 4, "tensor has to have 4 dimensions"
            if not (tensor.size(1) == 1 or tensor.size(1) == 3):
                warnings.warn("The 1. dimension (channel) has to be either 1 (gray) or 3 (rgb), taking the first "
                              "dimension now !!!")
                tensor = tensor[:, 0:1, ]

            grid = make_grid(tensor, **image_args)
            image = grid.mul(255).clamp(0, 255).byte().numpy()
        elif isinstance(tensor, np.ndarray):
            grid = np_make_grid(tensor, **image_args)
            image = np.clip(grid * 255, a_min=0, a_max=255)
            image = image.astype(np.uint8)

        else:
            raise ValueError("Tensor has to be a torch tensor or a numpy array")

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


    @convert_params
    @add_to_queue
    def show_image_grid_heatmap(self, heatmap, background=None, ratio=0.3, colormap=cm.jet,
                                  normalize=True, name="heatmap", caption=None,
                                  env_appendix="", opts=None, image_args=None, **kwargs):
        """
        Creates heat map from the given map and if given combines it with the background and then
        displays results with as image grid.

        Args:
           heatmap:  4d- tensor (N, C, H, W), if C = 3, colormap won't be applied.
           background: 4d- tensor (N, C, H, W)
           name: The name of the window
           ratio: The ratio to mix the map with the background (0 = only background, 1 = only map)
           caption: Caption of the generated image grid
           env_appendix: appendix to the environment name, if used the new env is env+env_appendix
           opts: opts dict for the ploty/ visdom plot, i.e. can set window size, en/disable ticks,...
           image_args: Arguments for the tensorvision save image method

        """

        if opts is None:
            opts = {}
        if image_args is None:
            image_args = {}
        if "normalize" not in image_args:
            image_args["normalize"] = normalize

        # if len(heatmap.shape) != 4:
        #     raise IndexError("'heatmap' must have dimensions BxCxHxW!")

        map_grid = np_make_grid(heatmap, normalize=normalize)  # map_grid.shape is (3, X, Y)
        if heatmap.shape[1] != 3:
            map_ = colormap(map_grid[0])[..., :-1].transpose(2, 0, 1)
        else:  # heatmap was already RGB, so don't apply colormap
            map_ = map_grid

        if background is not None:
            img_grid = np_make_grid(background, **image_args)
            fuse_img = (1.0 - ratio) * img_grid + ratio * map_
        else:
            fuse_img = map_

        fuse_img = np.clip(fuse_img * 255, a_min=0, a_max=255).astype(np.uint8)

        opts = opts.copy()
        opts.update(dict(
            title=name,
            caption=caption
        ))

        win = self.vis.image(
            img=fuse_img,
            win=name,
            env=self.name + env_appendix,
            opts=opts
        )

        return win


    @convert_params
    def show_embedding(self, tensor, labels=None, name=None, method="tsne", n_dims=2, n_neigh=30, meth_args=None,
                       *args, **kwargs):
        """
        Displays a tensor a an embedding

        Args:
            tensor: Tensor to be embedded an then displayed
            labels: Labels of the entries in the tensor (first dimension)
            name: The name of the window
            method: Method used for embedding, options are: tsne, standard, ltsa, hessian, modified, isomap, mds,
            spectral, umap
            n_dims: dimensions to embed the data into
            n_neigh: Neighbour parameter to kind of determin the embedding (see t-SNE for more information)
            meth_args: Further arguments which can be passed to the embedding method

        """

        if meth_args is None:
            meth_args = {}

        def __show_embedding(queue, tensor, labels=None, name=None, method="tsne", n_dims=2, n_neigh=30, **meth_args):
            emb_data = get_tensor_embedding(tensor, method=method, n_dims=n_dims, n_neigh=n_neigh, **meth_args)

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
        atexit.register(p.terminate)
        p.start()

    @convert_params
    def show_roc_curve(self, tensor, labels, name, reduce_to_n_samples=None, use_sub_process=False):
        """
        Displays a roc curve given a tensor with scores and the coresponding labels

        Args:
            tensor: Tensor with scores (e.g class probability )
            labels: Labels of the samples to which the scores match
            name: The name of the window
            reduce_to_n_samples: Reduce/ downsample to to n samples for fewer data points
            use_sub_process: Use a sub process to do the processing

        """

        res_fn = lambda tpr, fpr: self.show_lineplot(tpr, fpr, name=name, opts={"fillarea": True,
                                                                                "webgl": True})
        PytorchExperimentLogger.get_roc_curve(tensor=tensor, labels=labels, reduce_to_n_samples=reduce_to_n_samples,
                                              use_sub_process=use_sub_process, results_fn=res_fn)

    @convert_params
    def show_pr_curve(self, tensor, labels, name, reduce_to_n_samples=None, use_sub_process=False):
        """
        Displays a precision recall curve given a tensor with scores and the coresponding labels

        Args:
            tensor: Tensor with scores (e.g class probability )
            labels: Labels of the samples to which the scores match
            name: The name of the window
            reduce_to_n_samples: Reduce/ downsample to to n samples for fewer data points
            use_sub_process: Use a sub process to do the processing
        """
        res_fn = lambda precision, recall: self.show_lineplot(precision, recall, name=name, opts={"fillarea": True,
                                                                                                  "webgl": True})
        PytorchExperimentLogger.get_pr_curve(tensor=tensor, labels=labels, reduce_to_n_samples=reduce_to_n_samples,
                                             use_sub_process=use_sub_process, results_fn=res_fn)

    @convert_params
    def show_classification_metrics(self, tensor, labels, name, metric=("roc-auc", "pr-score"),
                                    use_sub_process=False, tag_name=None):
        """
        Displays some classification metrics as line plots in a graph (similar to show value (also uses show value
        for the caluclated values))

        Args:
            tensor: Tensor with scores (e.g class probability )
            labels: Labels of the samples to which the scores match
            name: The name of the window
            metric: List of metrics to calculate. Options are: roc-auc, pr-auc, pr-score, mcc, f1

        Returns:

        """

        res_fn = lambda val, name, tag: self.show_value(val, name=name, tag=tag)
        PytorchExperimentLogger.get_classification_metrics(tensor=tensor, labels=labels, name=name, metric=metric,
                                                           use_sub_process=use_sub_process, tag_name=tag_name,
                                                           results_fn=res_fn)

    def show_image_gradient(self, model, inpt, err_fn, grad_type="vanilla", n_runs=20, eps=0.1,
                            abs=False, **image_grid_params):
        """
        Given a model creates calculates the error and backpropagates it to the image and saves it (saliency map).

        Args:
            model: The model to be evaluated
            inpt: Input to the model
            err_fn: The error function the evaluate the output of the model on
            grad_type: Gradient calculation method, currently supports (vanilla, vanilla-smooth, guided,
            guided-smooth) ( the guided backprob can lead to segfaults -.-)
            n_runs: Number of runs for the smooth variants
            eps: noise scaling to be applied on the input image (noise is drawn from N(0,1))
            abs (bool): Flag, if the gradient should be a absolute value
            **image_grid_params: Params for make image grid.

        """
        grad = PytorchExperimentLogger.get_input_gradient(model=model, inpt=inpt, err_fn=err_fn, grad_type=grad_type,
                                                          n_runs=n_runs, eps=eps, abs=abs)
        self.show_image_grid(grad, **image_grid_params)

    def show_video(self, frame_list=None, name="frames", dim="LxHxWxC", scale=1.0, fps=25):
        self.vis.video(tensor=np.array(frame_list), dim=dim, opts={'fps': fps})
