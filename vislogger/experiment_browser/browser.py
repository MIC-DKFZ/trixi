import argparse
import json
import os

from flask import Flask, render_template, request, Blueprint

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
            keys.update(list(exp.keys()))
            exps.append(exp)

    ### Remove unwanted keys
    keys -= set(IGNORE_KEYS)

    ### Generate table rows
    sorted_keys = sorted(keys, key=lambda x: str(x).lower())

    rows = []
    for exp in exps:
        sub_row = []
        for key in sorted_keys:
            attr_strng = str(getattr(exp, key, "----"))
            sub_row.append((attr_strng, attr_strng[:25]))
        rows.append((os.path.basename(exp.work_dir),
                    str(getattr(exp, "experiment_name", "----")),
                    str(getattr(exp, "init_time", "----")),
                    str(getattr(exp, "", "----")),
                    sub_row))



    return {"cols": sorted_keys, "rows": rows}

def get_experiment_content(experiment_dir):

    exp = Experiment(experiment_dir)

    images = exp.get_images()
    plots = exp.get_plots()
    models = exp.get_checkpoints()

    return {'images': images, 'plots': plots, 'models': models}

@app.route("/")
def overview():
    base_info = process_base_dir(base_dir)
    return render_template("overview.html", title=base_dir, **base_info)

@app.route('/experiment/', methods=["GET"])
def experiment():

    id_ = request.args.get('id')
    if id_ is None:
        return "ERR: Please give an experiment ID"

    experiment_dir = os.path.join(base_dir, id_)

    if not os.path.exists(experiment_dir) or not os.path.isdir(experiment_dir):
        return "ERR: Please give a valid experiment directory"

    experiment_content = get_experiment_content(experiment_dir)
    return render_template('experiment.html', title=id_, **experiment_content)

if __name__ == "__main__":
    app.run()
