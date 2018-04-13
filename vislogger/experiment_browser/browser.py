import argparse
import numpy as np
import os
from scipy.signal import savgol_filter

from flask import Flask, render_template, request, Blueprint, Markup, abort
from plotly.offline import plot
import plotly.graph_objs as go
import colorlover as cl

from vislogger.experiment_browser.experiment import Experiment



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
    results = exp.get_results()
    graphs = make_graphs(results)
    images = exp.get_images()

    return {"graphs": graphs, "images": images}



def make_graphs(results, trace_options=None, layout_options=None):

    if trace_options is None: trace_options = {}
    if layout_options is None: layout_options = {}

    graphs = []

    for group in results:

        layout = go.Layout(title=group, **layout_options)
        traces = []

        for r, result in enumerate(results[group]):

            y = np.array(results[group][result]["data"])
            x = np.array(results[group][result]["epoch"])

            do_filter = len(y) >= 1000
            opacity = 0.2 if do_filter else 1.

            traces.append(go.Scatter(x=x, y=y, opacity=opacity, name=result,
                                     line=dict(color=COLORMAP[r%len(COLORMAP)]), **trace_options))
            if do_filter:
                filter_ = lambda x: savgol_filter(x, max(5, 2 * (len(y) // 50) + 1), 3)
                traces.append(go.Scatter(x=x, y=filter_(y), name=result+" smoothed",
                                         line=dict(color=COLORMAP[r%len(COLORMAP)]), **trace_options))

        graphs.append(Markup(plot({"data": traces, "layout": layout},
                             output_type="div",
                             include_plotlyjs=False,
                             show_link=False)))

    return graphs



@app.route("/")
def overview():

    try:
        base_info = process_base_dir(base_dir)
        base_info["title"] = base_dir
        return render_template("overview.html", **base_info)
    except Exception as e:
        print(e.__repr__())
        abort(500)

@app.route('/experiment/', methods=['GET'])
def experiment():

    experiments = request.args.getlist('exp')

    results = []
    for experiment in experiments:
        exp = Experiment(os.path.join(base_dir, experiment))
        results.append(exp.get_results())
    # results = merge_results(results)

    content = {}
    # content["graphs"] = make_graphs(results)
    content["title"] = experiments

    return render_template('experiment.html', **content)


@app.route('/experiment/<experiment_name>', methods=['GET'])
def experiment_(experiment_name):

    experiment_dir = os.path.join(base_dir, experiment_name)
    if not os.path.exists(experiment_dir) or not os.path.isdir(experiment_dir):
        print("ERR: Please give a valid experiment directory")
        abort(404)

    experiment_content = get_experiment_content(experiment_dir)
    experiment_content["title"] = experiment_name

    return render_template('experiment.html', **experiment_content)

if __name__ == "__main__":
    app.run(debug=True)
