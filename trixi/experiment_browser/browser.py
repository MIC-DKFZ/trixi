import argparse
import json
import os
import sys
from collections import OrderedDict

import colorlover as cl
from flask import Blueprint, Flask, abort, render_template, request

from trixi.experiment_browser.dataprocessing import group_images, make_graphs, merge_results, process_base_dir
from trixi.experiment_browser.experimentreader import ExperimentReader, CombiExperimentReader, group_experiments_by
from trixi.util import Config

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


def parse_args():
    # Read in base directory

    if len(sys.argv) > 1:

        parser = argparse.ArgumentParser()
        parser.add_argument("base_directory",
                            help="Give the path to the base directory of your project files",
                            type=str, default=".")
        parser.add_argument("-d", "--debug", action="store_true",
                            help="Turn debug mode on, eg. for live reloading.")
        parser.add_argument("-x", "--expose", action="store_true",
                            help="Make server externally visible")
        parser.add_argument("-p", "--port", default=5000, type=int,
                            help="Port to start the server on (5000 by default)")
        args = parser.parse_args()
        base_dir = args.base_directory
        if base_dir[-1] == os.sep:
            base_dir = base_dir[:-1]

    else:
        args = type('', (), {})()
        args.port = 5000
        args.expose = False
        args.debug = False
        args.base_directory = "."
        base_dir = "."

    return args, base_dir


def create_flask_app(base_dir):
    # The actual flask app lives in the package directory. The blueprint allows us
    # to specify an additional static folder and we use that to give access to the
    # experiment files
    app = Flask(__name__, static_folder=os.path.join(os.path.dirname(__file__), "static"))
    blueprint = Blueprint("data", __name__, static_url_path=base_dir, static_folder=base_dir)
    app.register_blueprint(blueprint)

    return app


def register_url_routes(app, base_dir):
    app.add_url_rule("/", "overview", lambda: overview(base_dir), methods=["GET"])
    app.add_url_rule("/overview", "overview_", lambda: overview_(base_dir), methods=["GET"])
    app.add_url_rule('/experiment', "experiment", lambda: experiment(base_dir), methods=['GET'])
    app.add_url_rule('/combine', "combine", lambda: combine(base_dir), methods=['GET'])
    app.add_url_rule('/experiment_log', "experiment_log", lambda: experiment_log(base_dir), methods=['GET'])
    app.add_url_rule('/experiment_plots', "experiment_plots", lambda: experiment_plots(base_dir), methods=['GET'])
    app.add_url_rule('/experiment_remove', "experiment_remove", lambda: experiment_remove(base_dir), methods=['GET'])
    app.add_url_rule('/experiment_star', "experiment_star", lambda: experiment_star(base_dir), methods=['GET'])
    app.add_url_rule('/experiment_rename', "experiment_rename", lambda: experiment_rename(base_dir), methods=['GET'])


def start_browser():
    args, base_dir = parse_args()

    base_dir = os.path.abspath(base_dir)

    app = create_flask_app(base_dir)
    register_url_routes(app, base_dir)

    host = "0.0.0.0" if args.expose else "localhost"
    app.run(debug=args.debug, host=host, port=args.port)


def overview(base_dir):
    try:
        base_info = process_base_dir(base_dir, ignore_keys=IGNORE_KEYS)
        base_info["title"] = base_dir
        return render_template("overview.html", **base_info)
    except Exception as e:
        print(e.__repr__())
        raise e
        abort(500)


def overview_(base_dir):
    dir_ = request.args.get("dir")
    try:
        base_info = process_base_dir(base_dir, dir_, ignore_keys=IGNORE_KEYS)
        base_info["title"] = dir_
        return render_template("overview.html", **base_info)
    except Exception as e:
        print(e.__repr__())
        raise e
        abort(500)


def experiment(base_dir):
    experiment_paths = request.args.getlist('exp')
    name = request.args.get('name', "")
    do_save = request.args.get('save', "")
    combi = request.args.get('combi', 'false')

    experiments = []

    if combi == "true":
        combi_exp = CombiExperimentReader(base_dir, experiment_paths, name=name)
        if do_save == "true":
            combi_exp.save()
        experiments = [combi_exp]
    else:
        # Get all Experiments
        for experiment_path in sorted(experiment_paths):
            exp = ExperimentReader(base_dir, experiment_path)
            experiments.append(exp)

    # Assign unique names
    exp_names = [exp.exp_name for exp in experiments]
    if len(exp_names) > len(set(exp_names)):
        for i, exp in enumerate(experiments):
            exp.exp_name += str(i)
    exp_names = [exp.exp_name for exp in experiments]

    # Site Content
    content = {}

    # Get config
    default_val = "-"
    combi_config = {}
    exp_configs = [exp.config.flat(False) for exp in experiments]
    diff_config_keys = list(Config.difference_config_static(*exp_configs).flat().keys())

    config_keys = set([k for c in exp_configs for k in c.keys()])
    for k in sorted(config_keys):
        combi_config[k] = []
        for conf in exp_configs:
            combi_config[k].append(conf.get(k, default_val))
    config_keys = list(sorted(list(config_keys)))

    # Get results
    default_val = "-"
    combi_results = {}
    exp_results = [exp.get_results() for exp in experiments]
    result_keys = set([k for r in exp_results for k in r.keys()])
    for k in sorted(result_keys):
        combi_results[k] = []
        for res in exp_results:
            combi_results[k].append(res.get(k, default_val))
    result_keys = list(sorted(list(result_keys)))

    # Get images
    images = OrderedDict({})
    image_keys = set()
    image_path = {}
    for exp in experiments:
        exp_images = exp.get_images()
        img_groups = group_images(exp_images)
        images[exp.exp_name] = img_groups
        image_path[exp.exp_name] = exp.img_dir
        image_keys.update(list(img_groups.keys()))
    image_keys = list(image_keys)
    image_keys.sort()

    # Get Plots
    plots = OrderedDict({})
    for exp in experiments:
        exp_plots = exp.get_plots()
        plots[exp.exp_name] = exp_plots

    # Get logs
    logs_dict = OrderedDict({})
    for exp in experiments:
        exp_logs = [(os.path.basename(l), exp.exp_dir) for l in exp.get_logs()]
        logs_dict[exp.exp_name] = exp_logs

    content["title"] = experiments
    content["images"] = {"img_path": image_path, "imgs": images, "img_keys": image_keys}
    content["plots"] = {"plots": plots}
    content["config"] = {"exps": experiments, "configs": combi_config, "keys": config_keys,
                         "diff_keys": diff_config_keys}
    content["results"] = {"exps": exp_names, "results": combi_results, "keys": result_keys}
    content["logs"] = {"logs_dict": logs_dict}

    return render_template('experiment.html', **content)


def experiment_log(base_dir):
    experiment_path = request.args.get('exp')
    log_name = request.args.get('log')

    exp = ExperimentReader(base_dir, experiment_path)
    content = exp.get_log_file_content(log_name)

    print(experiment_path, log_name)

    return content


def experiment_remove(base_dir):
    experiment_paths = request.args.getlist('exp')

    # Get all Experiments
    for experiment_path in sorted(experiment_paths):
        exp = ExperimentReader(base_dir, experiment_path)
        exp.ignore_experiment()

    return ""


def experiment_plots(base_dir):
    experiment_paths = request.args.getlist('exp')
    combi = request.args.get('combi', 'false')
    experiments = []

    if combi == "true":
        combi_exp = CombiExperimentReader(base_dir, experiment_paths)
        # combi_exp.save()
        experiments = [combi_exp]
    else:
        # Get all Experiments
        for experiment_path in sorted(experiment_paths):
            exp = ExperimentReader(base_dir, experiment_path)
            experiments.append(exp)

    # Assign unique names
    exp_names = [exp.exp_name for exp in experiments]
    if len(exp_names) > len(set(exp_names)):
        for i, exp in enumerate(experiments):
            exp.exp_name += str(i)
    exp_names = [exp.exp_name for exp in experiments]

    results = []
    for exp in experiments:
        results.append(exp.get_results_log())
    results = merge_results(exp_names, results)

    graphs, traces = make_graphs(results, color_map=COLORMAP)
    graphs = [str(g) for g in graphs]

    return json.dumps({"graphs": graphs, "traces": traces})


def experiment_star(base_dir):
    experiment_path = request.args.get('exp')
    star_val = request.args.get('star')

    star_val = bool(int(star_val))

    print(experiment_path, star_val)

    exp = ExperimentReader(base_dir, experiment_path)
    exp.update_meta_info(star=star_val)

    return ""


def experiment_rename(base_dir):
    experiment_path = request.args.get('exp')
    new_name = request.args.get('name')

    print(experiment_path, new_name)

    exp = ExperimentReader(base_dir, experiment_path)
    exp.update_meta_info(name=new_name)

    return ""


def combine(base_dir):
    experiment_paths = request.args.getlist('exp')
    group_by_list = request.args.getlist('group')
    name = request.args.get('name', "")

    try:
        exps = []
        for ep in experiment_paths:
            expr = ExperimentReader(os.path.join(base_dir, ep))
            if expr.exp_info.get("state", "Combined") != "Combined":
                exps.append(expr)

        exp_groups = group_experiments_by(exps, group_by_list)

        combis = []
        for group in exp_groups:
            expc = CombiExperimentReader(base_dir="", exp_dirs=[e.work_dir for e in group], name=name)
            combis.append(expc)

        for expc in combis:
            expc.save()

        return "1"
    except Exception as e:
        return "0"

    return "0"


if __name__ == "__main__":
    start_browser()
