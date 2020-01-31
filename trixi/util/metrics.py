import atexit
import warnings
from collections import OrderedDict
from multiprocessing import Process

import numpy as np
from sklearn import metrics


def get_roc_curve(tensor, labels, reduce_to_n_samples=None, use_sub_process=False, results_fn=lambda
        x, *y, **z: None):
    """
    Displays a roc curve given a tensor with scores and the coresponding labels

    Args:
        tensor: Tensor with scores (e.g class probability )
        labels: Labels of the samples to which the scores match
        reduce_to_n_samples: Reduce/ downsample to to n samples for fewer data points
        use_sub_process: Use a sub process to do the processing, if true nothing is returned
        results_fn: function which is called with the results/ return values. Expected f(tpr, fpr)

    """

    def __get_roc_curve(tensor, labels, reduce_to_n_samples=None, results_fn=lambda x, *y, **z: None):

        if not isinstance(labels, list):
            labels = labels.flatten()
        if not isinstance(tensor, list):
            tensor = tensor.flatten()

        fpr, tpr, thresholds = metrics.roc_curve(labels, tensor)
        if reduce_to_n_samples is not None:
            fpr = [np.mean(x) for x in np.array_split(fpr, reduce_to_n_samples)]
            tpr = [np.mean(x) for x in np.array_split(tpr, reduce_to_n_samples)]
        results_fn(tpr, fpr)

        return tpr, fpr
        # self.show_lineplot(tpr, fpr, name=name, opts={"fillarea": True, "webgl": True})
        # self.add_to_graph(x_vals=np.arange(0, 1.1, 0.1), y_vals=np.arange(0, 1.1, 0.1), name=name, append=True)

    if use_sub_process:
        p = Process(target=__get_roc_curve, kwargs=dict(tensor=tensor,
                                                        labels=labels,
                                                        reduce_to_n_samples=reduce_to_n_samples,
                                                        results_fn=results_fn
                                                        ))
        atexit.register(p.terminate)
        p.start()
    else:
        try:
            return __get_roc_curve(tensor=tensor,
                                   labels=labels,
                                   reduce_to_n_samples=reduce_to_n_samples,
                                   results_fn=results_fn
                                   )
        except Exception as e:
            warnings.warn("Sth went wrong with calculating the roc curve")


def get_pr_curve(tensor, labels, reduce_to_n_samples=None, use_sub_process=False,
                 results_fn=lambda x, *y, **z: None):
    """
    Displays a precision recall curve given a tensor with scores and the coresponding labels

    Args:
        tensor: Tensor with scores (e.g class probability )
        labels: Labels of the samples to which the scores match
        reduce_to_n_samples: Reduce/ downsample to to n samples for fewer data points
        use_sub_process: Use a sub process to do the processing, if true nothing is returned
        results_fn: function which is called with the results/ return values. Expected f(precision, recall)

    """

    def __get_pr_curve(tensor, labels, reduce_to_n_samples=None, results_fn=lambda x, *y, **z: None):

        if not isinstance(labels, list):
            labels = labels.flatten()
        if not isinstance(tensor, list):
            tensor = tensor.flatten()

        precision, recall, thresholds = metrics.precision_recall_curve(labels, tensor)
        if reduce_to_n_samples is not None:
            precision = [np.mean(x) for x in np.array_split(precision, reduce_to_n_samples)]
            recall = [np.mean(x) for x in np.array_split(recall, reduce_to_n_samples)]
        results_fn(precision, recall)

        return precision, recall
        # self.show_lineplot(precision, recall, name=name, opts={"fillarea": True, "webgl": True})
        # self.add_to_graph(x_vals=np.arange(0, 1.1, 0.1), y_vals=np.arange(0, 1.1, 0.1), name=name, append=True)

    if use_sub_process:
        p = Process(target=__get_pr_curve, kwargs=dict(tensor=tensor,
                                                       labels=labels,
                                                       reduce_to_n_samples=reduce_to_n_samples,
                                                       results_fn=results_fn
                                                       ))
        atexit.register(p.terminate)
        p.start()
    else:
        try:
            return __get_pr_curve(tensor=tensor,
                                  labels=labels,
                                  reduce_to_n_samples=reduce_to_n_samples,
                                  results_fn=results_fn
                                  )
        except Exception as e:
            warnings.warn("Sth went wrong with calculating the pr curve")


def _get_classification_metrics(tensor, labels, metric="roc-auc"):
    """
   Calculates a metric given the predicted values and the given/correct labels.

    Args:
        tensor: Tensor with scores (e.g class probability )
        labels: Labels of the samples to which the scores match
        metric: List of metrics to calculate. Options are: roc-auc, pr-auc, pr-score, mcc, f1

    Returns:
        The metric value

    """

    if not isinstance(labels, list):
        labels = labels.flatten()
    if not isinstance(tensor, list):
        tensor = tensor.flatten()

    metric_value = 0.0
    if "roc-auc" == metric:
        metric_value = metrics.roc_auc_score(labels, tensor)
    if "pr-auc" == metric:
        precision, recall, thresholds = metrics.precision_recall_curve(labels, tensor)
        metric_value = metrics.auc(recall, precision)
    if "pr-score" == metric:
        metric_value = metrics.average_precision_score(labels, tensor)
    if "mcc" == metric:
        metric_value = metrics.matthews_corrcoef(labels, tensor)
    if "f1" == metric:
        metric_value = metrics.f1_score(labels, tensor)

    return metric_value


def get_classification_metrics(tensor, labels, name="", metric=("roc-auc", "pr-score"), use_sub_process=False,
                               tag_name=None, results_fn=lambda x, *y, **z: None):
    """
    Displays some classification metrics as line plots in a graph (similar to show value (also uses show value
    for the caluclated values))

    Args:
        tensor: Tensor with scores (e.g class probability )
        labels: Labels of the samples to which the scores match
        name: The name of the window
        metric: List of metrics to calculate. Options are: roc-auc, pr-auc, pr-score, mcc, f1
        tag_name: Name for the tag, if no given use name
        use_sub_process: Use a sub process to do the processing, if true nothing is returned
        results_fn: function which is called with the results/ return values. Expected f(val, name, tag)

    Returns:

    """

    def __get_classification_metrics(tensor, labels, name="", metric=("roc-auc", "pr-score"),
                                     tag_name=None, results_fn=lambda x, *y, **z: None):

        if not isinstance(labels, list):
            labels = labels.flatten()
        if not isinstance(tensor, list):
            tensor = tensor.flatten()

        res_dict = OrderedDict()

        for m in metric:
            res_dict[m] = _get_classification_metrics(tensor, labels, m)

        for tag, val in res_dict.items():
            results_fn(val, name=tag + "-" + name, tag=tag_name)

        return res_dict

    if use_sub_process:
        p = Process(target=__get_classification_metrics, kwargs=dict(tensor=tensor,
                                                                     labels=labels,
                                                                     name=name,
                                                                     metric=metric,
                                                                     tag_name=tag_name,
                                                                     results_fn=results_fn
                                                                     ))
        atexit.register(p.terminate)
        p.start()
    else:
        try:
            return __get_classification_metrics(tensor=tensor,
                                                labels=labels,
                                                name=name,
                                                metric=metric,
                                                tag_name=tag_name,
                                                results_fn=results_fn
                                                )

        except Exception as e:
            warnings.warn("Sth went wrong with calculating the classification metrics")
            ret = {m: 0.0 for m in metric}
            return ret


def get_classification_metric(tensor, labels, metric="roc-auc", use_sub_process=False):
    """
   Calculates a metric given the predicted values and the given/correct labels.

    Args:
        tensor: Tensor with scores (e.g class probability )
        labels: Labels of the samples to which the scores match
        metric: List of metrics to calculate. Options are: roc-auc, pr-auc, pr-score, mcc, f1
        use_sub_process: Use a sub process to do the processing, if true nothing is returned

    Returns:
        The metric value

    """

    if use_sub_process:
        p = Process(target=_get_classification_metrics, kwargs=dict(tensor=tensor,
                                                                    labels=labels,
                                                                    metric=metric,
                                                                    ))
        atexit.register(p.terminate)
        p.start()
    else:
        try:
            return _get_classification_metrics(tensor=tensor,
                                               labels=labels,
                                               metric=metric,
                                               )

        except Exception as e:
            warnings.warn("Sth went wrong with calculating the classification metrics")
            return 0.0
