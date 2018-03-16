import argparse
import json
import numpy as np
import os
from scipy.ndimage.filters import gaussian_filter1d

from flask import Flask, render_template, request, Blueprint, Markup
from plotly.offline import plot
import plotly.graph_objs as go

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

    #images = exp.get_images()
    #plots = exp.get_plots()
    #models = exp.get_checkpoints()

    results, spacings = exp.get_results()

    graphs = []
    for key in results:
        if np.issubdtype(results[key].dtype, np.number):
            graphs.append(make_graph(key, results[key], spacings[key], smoothing=100.))

    return {"graphs": graphs}

    #return {'images': images, 'plots': plots, 'models': models}

def make_graph(name, y, x_scale=1., trace_options=None, smoothing=False):

    traces = []
    x = np.arange(1, y.shape[0]+1) * x_scale
    if trace_options is None: trace_options = {}

    if smoothing:
        filter_ = lambda x: gaussian_filter1d(x, smoothing)
    else:
        filter_ = lambda x: x

    if y.ndim == 1:
        traces.append(go.Scatter(x=x, y=filter_(y), **trace_options))

    elif y.ndim == 2:
        for t in range(y.shape[1]):
            traces.append(go.Scatter(x=x, y=filter_(y[:,t]), **trace_options))

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
    base_info = process_base_dir(base_dir)
    return render_template("overview.html", title=base_dir, **base_info)

@app.route('/experiment/<experiment_name>', methods=['GET'])
def experiment(experiment_name):
    experiment_dir = os.path.join(base_dir, experiment_name)

    if not os.path.exists(experiment_dir) or not os.path.isdir(experiment_dir):
        return "ERR: Please give a valid experiment directory"

    experiment_content = get_experiment_content(experiment_dir)
    return render_template('experiment.html', title=experiment_name, **experiment_content)

if __name__ == "__main__":
    app.run()
