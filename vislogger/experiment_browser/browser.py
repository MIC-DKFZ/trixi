import argparse
import numpy as np
import os
from scipy.signal import savgol_filter

from flask import Flask, render_template, request, Blueprint, Markup, abort
from plotly.offline import plot
import plotly.graph_objs as go
import colorlover as cl

from .experiment import Experiment

IGNORE_KEYS = ("experiment_name",
               "experiment_dir",
               "work_dir",
               "config_dir",
               "log_dir",
               "checkpoint_dir",
               "img_dir",
               "plot_dir",
               "save_dir",
               "result_dir",
               "init_time",
               "note")

COLORMAP = cl.scales["8"]["qual"]["Dark2"]

### Read in base directory
parser = argparse.ArgumentParser()
parser.add_argument("base_directory",
                    help="Give the path to the base directory of your project files",
                    type=str)
args = parser.parse_args()
base_dir = args.base_directory

app = Flask(__name__, static_folder=os.path.join(os.path.dirname(__file__), "static"))
blueprint = Blueprint("data", __name__, static_url_path=base_dir, static_folder=base_dir)
app.register_blueprint(blueprint)

def process_base_dir(base_dir):

    keys = set()
    exps = []

    ### Load Experiments with keys / different param values
    for sub_dir in sorted(os.listdir(base_dir)):
        dir_path = os.path.join(base_dir, sub_dir)
        if os.path.isdir(dir_path):

            exp = Experiment(dir_path)
            keys.update(list(exp.config.keys()))
            exps.append(exp)

    ### Remove unwanted keys
    keys -= set(IGNORE_KEYS)

    ### Generate table rows
    sorted_keys = sorted(keys, key=lambda x: str(x).lower())

    rows = []
    for exp in exps:
        sub_row = []
        for key in sorted_keys:
            attr_strng = str(getattr(exp.config, key, "----"))
            sub_row.append((attr_strng, attr_strng[:25]))
        rows.append((os.path.basename(exp.work_dir),
                    str(getattr(exp.config, "experiment_name", "----")),
                    str(getattr(exp.config, "init_time", "----")),
                    str(getattr(exp.config, "note", "----")),
                    sub_row))

    return {"cols": sorted_keys, "rows": rows}

def get_experiment_content(experiment_dir):

    exp = Experiment(experiment_dir)
    results, _ = exp.get_results()

    graphs = []
    for key in results:
        if np.issubdtype(results[key].dtype, np.number):
            graphs.append(make_graph(key, results[key]))

    return {"graphs": graphs}

    #return {'images': images, 'plots': plots, 'models': models}

def make_graph(name, y, x=None, labels=None, trace_options=None, layout_options=None):

    print(name, y.shape)

    if x is None: x = np.arange(1, y.shape[0]+1)
    if trace_options is None: trace_options = {}
    if layout_options is None: layout_options = {}

    filter_ = lambda x: savgol_filter(x, 2 * (len(y) // 200) + 1, 3)

    traces = []

    if y.ndim == 1:

        if labels is None:
            labels = ["Data", "Smoothed"]
        else:
            if isinstance(labels, str):
                labels = [labels, labels + " smoothed"]
            else:
                labels = [labels[0], labels[0] + " smoothed"]

        traces.append(go.Scatter(x=x, y=y, opacity=0.2, name=labels[0], **trace_options,
                                 line=dict(color=COLORMAP[0])))
        traces.append(go.Scatter(x=x, y=filter_(y), name=labels[1], **trace_options,
                                 line=dict(color=COLORMAP[0])))

    elif y.ndim == 2:

        if labels is None: labels = [str(x) for x in range(y.shape[1])]
        labels = [labels[i//2] if i%2==0 else labels[i//2] + " smoothed" for i in range(2*len(labels))]

        for t in range(y.shape[1]):

            traces.append(go.Scatter(x=x, y=y[:,t],opacity=0.2, name=labels[2*t], **trace_options,
                                     line=dict(color=COLORMAP[t])))
            traces.append(go.Scatter(x=x, y=filter_(y[:,t]), name=labels[2*t+1], **trace_options,
                                     line=dict(color=COLORMAP[t])))

    else:
        print("ERR: can only create plots for arrays with ndim <= 3.")
        return ""

    layout = go.Layout(title=name)
    markup = plot({"data": traces, "layout": layout},
                  output_type="div",
                  include_plotlyjs=False,
                  show_link=False)

    return Markup(markup)

@app.route("/")
def overview():

    try:
        base_info = process_base_dir(base_dir)
        base_info["title"] = base_dir
        return render_template("overview.html", **base_info)
    except Exception as e:
        print(e.__repr__())
        abort(500)

@app.route('/experiment/<experiment_name>', methods=['GET'])
def experiment(experiment_name):

    experiment_dir = os.path.join(base_dir, experiment_name)
    if not os.path.exists(experiment_dir) or not os.path.isdir(experiment_dir):
        print("ERR: Please give a valid experiment directory")
        abort(404)

    smooth = request.args.getlist("smooth")

    experiment_content = get_experiment_content(experiment_dir)
    experiment_content["title"] = experiment_name

    return render_template('experiment.html', **experiment_content)

if __name__ == "__main__":
    app.run()
