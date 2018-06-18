import os
from collections import defaultdict
import re

import colorlover as cl
import numpy as np
import plotly.graph_objs as go
from flask import Markup
from plotly.offline import plot
from scipy.signal import savgol_filter

from trixi.experiment_browser.experimentreader import ExperimentReader

# These keys will be ignored when in a config file
IGNORE_KEYS = ("name",
               "experiment_dir",
               "work_dir",
               "config_dir",
               "log_dir",
               "checkpoint_dir",
               "img_dir",
               "plot_dir",
               "save_dir",
               "result_dir",
               "time",
               "state")

# Set the color palette for plots
COLORMAP = cl.scales["8"]["qual"]["Dark2"]


def process_base_dir(base_dir, view_dir="", default_val="-", short_len=25, ignore_keys=IGNORE_KEYS):
    """Create an overview table of all experiments in the given directory.

    Args:
        directory (str): A directory containing experiment folders.
        default_val (str): Default value if an entry is missing.
        short_len (int): Cut strings to this length. Full string in alt-text.

    Returns:
        dict: {"ccols": Columns for config entries,
               "rcols": Columns for result entries,
               "rows": The actual data}

    """

    full_dir = os.path.join(base_dir, view_dir)

    config_keys = set()
    result_keys = set()
    exps = []
    non_exps = []

    ### Load Experiments with keys / different param values
    for sub_dir in sorted(os.listdir(full_dir)):
        dir_path = os.path.join(full_dir, sub_dir)
        if os.path.isdir(dir_path):
            try:
                exp = ExperimentReader(full_dir, sub_dir)
                if exp.ignore:
                    continue
                config_keys.update(list(exp.config.keys()))
                result_keys.update(list(exp.get_results().keys()))
                exps.append(exp)
            except Exception as e:
                print("Could not load experiment: ", dir_path)
                print(e)
                print("-" * 20)
                non_exps.append(os.path.join(view_dir, sub_dir))

    ### Remove unwanted keys
    config_keys -= set(ignore_keys)
    result_keys -= set(ignore_keys)

    ### Generate table rows
    sorted_c_keys = sorted(config_keys, key=lambda x: str(x).lower())
    sorted_r_keys = sorted(result_keys, key=lambda x: str(x).lower())

    rows = []
    for exp in exps:
        config_row = []
        for key in sorted_c_keys:
            attr_strng = str(exp.config.get(key, default_val))
            config_row.append((attr_strng, attr_strng[:short_len]))
        result_row = []
        for key in sorted_r_keys:
            attr_strng = str(exp.get_results().get(key, default_val))
            result_row.append((attr_strng, attr_strng[:short_len]))

        name = exp.exp_name
        time = exp.exp_info.get("time", default_val) if "time" in exp.exp_info else exp.config.get("time", default_val)
        state = exp.exp_info.get("state", default_val) if "state" in exp.exp_info else exp.config.get("state",
                                                                                                      default_val)

        rows.append((os.path.relpath(exp.work_dir, base_dir),
                     exp.star,
                     str(name),
                     str(time),
                     str(state),
                     config_row, result_row))

    return {"ccols": sorted_c_keys, "rcols": sorted_r_keys, "rows": rows, "noexp": non_exps}


def group_images(images):
    images.sort()
    group_dict = defaultdict(list)

    for img in images:
        filename = img.split(os.sep + "img" + os.sep)[1]
        base_name = os.path.splitext(filename)[0]
        number_groups = re.findall("\d+", base_name)
        if len(number_groups) == 0:
            base_name = ''.join(e for e in base_name if e.isalpha())
        else:
            base_name = base_name.replace(number_groups[0], "")

        group_dict[base_name].append(filename)

    return group_dict


def make_graphs(results, trace_options=None, layout_options=None, color_map=COLORMAP):
    """Create plot markups.

    This converts results into plotly plots in markup form. Results in a common
    group will be placed in the same plot.

    Args:
        results (dict): Dictionary

    """

    if trace_options is None:
        trace_options = {}
    if layout_options is None:
        layout_options = {
            "legend": dict(
                orientation="v",
                xanchor="left",
                x=0,
                yanchor="top",
                y=-0.1,
                font=dict(
                    size=8,
                )
            )
        }

    graphs = []
    trace_counters = []

    for group in sorted(results):

        layout = go.Layout(title=group, **layout_options)
        traces = []

        for r, result in enumerate(sorted(results[group])):

            y = np.array(results[group][result]["data"])
            x = np.array(results[group][result]["counter"])

            do_filter = len(y) >= 1000
            opacity = 0.2 if do_filter else 1.

            if do_filter:
                def filter_(x):
                    return savgol_filter(x, max(5, 2 * (len(y) // 50) + 1), 3)

                traces.append(go.Scatter(x=x, y=y, opacity=opacity, name=result, legendgroup=result, showlegend=False,
                                         line=dict(color=color_map[r % len(color_map)]), **trace_options))
                traces.append(go.Scatter(x=x, y=filter_(y), name=result, legendgroup=result,
                                         line=dict(color=color_map[r % len(color_map)]), **trace_options))
            else:
                traces.append(go.Scatter(x=x, y=y, opacity=opacity, name=result, legendgroup=result,
                                         line=dict(color=color_map[r % len(color_map)]), **trace_options))

        trace_counters.append(len(results[group]))
        graphs.append(Markup(plot({"data": traces, "layout": layout},
                                  output_type="div",
                                  include_plotlyjs=False,
                                  show_link=False)))

    return graphs, trace_counters


def merge_results(experiment_names, result_list):
    merged_results = {}

    for r, result in enumerate(result_list):
        for label in result.keys():
            if label not in merged_results:
                merged_results[label] = {}
            for key in result[label].keys():
                new_key = "_".join([experiment_names[r], key])
                merged_results[label][new_key] = result[label][key]

    return merged_results
