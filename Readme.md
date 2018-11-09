[![DOI](https://zenodo.org/badge/134823632.svg)](https://zenodo.org/badge/latestdoi/134823632)
[![PyPI version](https://badge.fury.io/py/trixi.svg)](https://badge.fury.io/py/trixi)
[![Build Status](https://travis-ci.org/MIC-DKFZ/trixi.svg?branch=master)](https://travis-ci.org/MIC-DKFZ/trixi)
[![Documentation Status](https://readthedocs.org/projects/trixi/badge/?version=latest)](https://trixi.readthedocs.io/en/latest/?badge=latest)
[![GitHub](https://img.shields.io/pypi/l/trixi.svg)](https://github.com/MIC-DKFZ/trixi/blob/master/LICENSE)
<p align="center">
    <img src="https://github.com/MIC-DKFZ/trixi/blob/master/doc/_static/logo/trixi-small.png">
</p>

Finally get some structure into your machine learning experiments.
*trixi* is a tool that helps you configure, log and visualize your experiments in a reproducible fashion.

* [Features](#features)
* [Installation](#installation)
* [Documentation](#documentation) ([trixi.rtfd.io](https://trixi.readthedocs.io/en/latest/))
* [Examples](#examples)

# Features

*trixi* consists of three parts:
* Logging API<br>
    *Log whatever data you like in whatever way you like to whatever backend you like.*

* Experiment Infrastructure<br>
    *Standardize your experiment, let the framework do all the inconvenient stuff, and simply start, resume,
    change and finetune all your experiments*.

* Experiment Browser <br>
    *Compare, combine and visually inspect the results of your experiments*.

An detailed implementation overview is given [here](https://trixi.readthedocs.io/en/latest/class_diagram.html).

### Logging API

The logging API provides a standardized way for logging results to different backends.
The logging API supports
(among others):
* Values
* Text
* Plots (Bar, Line, Scatter, Piechart, ...)
* Images (Single, Grid)

And offers different Backends, e.g. :
* Visdom ([visdom-loggers](https://trixi.readthedocs.io/en/latest/api/trixi.logger.visdom.html))
* Matplotlib / Seaborn ([plt-loggers](https://trixi.readthedocs.io/en/latest/api/trixi.logger.plt.html))
* Local Disk ([file-loggers](https://trixi.readthedocs.io/en/latest/api/trixi.logger.file.html))
* Telegram ([message-loggers](https://trixi.readthedocs.io/en/latest/api/trixi.logger.message.html))

And an [experiment-logger](https://trixi.readthedocs.io/en/latest/api/trixi.logger.experiment.experimentlogger.html) for logging your experiments, which uses a file logger to automatically create a structured directory and allows
storing of config, results, plots, dict, array, images, etc. That way your experiments will always have the same structure on disk.

Here are some examples:

* [Visdom](https://github.com/facebookresearch/visdom):<br>
<img src="https://lh3.googleusercontent.com/-h3HuvbU2V0SfgqgXGiK3LPghE5vqvS0pzpObS0YgG_LABMFk62JCa3KVu_2NV_4LJKaAa5-tg=s0" alt="visdom-logger" width="300"/>

* Files:<br>
<img src="https://github.com/MIC-DKFZ/trixi/blob/master/doc/_static/trixi_file.png" alt="file-logger" height="200"/>

* Telegram:<br>
<img src="https://github.com/MIC-DKFZ/trixi/blob/master/doc/_static/trixi_telegram.png" alt="telegram-logger" width="150"/>


### Experiment Infrastructure

The [Experiment Infrastructure](https://trixi.readthedocs.io/en/latest/api/trixi.experiment.pytorchexperiment.html) provides a unified way to configure, run, store and evaluate your results.
It gives you an Experiment interface, for which you can implement the training, validation and testing.
Furthermore it automatically provides you with easy access to the Logging API and stores your config as well as the
results for easy evaluation and reproduction. There is an abstract [Experiment](https://trixi.readthedocs.io/en/latest/api/trixi.experiment.experiment.html) class and a [PytorchExperiment](https://trixi.readthedocs.io/en/latest/api/trixi.experiment.pytorchexperiment.html) with many convenience features.

<img src="https://github.com/MIC-DKFZ/trixi/blob/master/doc/_static/trixi_exp2.png" alt="exp-train" height="300"/><img src="https://github.com/MIC-DKFZ/trixi/blob/master/doc/_static/trixi_exp1.png" alt="exp-test" height="300"/>

For more info, visit the Documentation.

### Experiment Browser
The Experiment Browser offers a complete overview of experiments along with all config parameters and results.
It also allows to combine and/or compare different experiments, giving you an interactive comparison highlighting differences in the configs and a detailed view of all images,
plots, results and logs of each experiment, with live plots and more.
![trixi browser](https://github.com/MIC-DKFZ/trixi/blob/master/doc/_static/trixi_browser.gif)

# Installation

Install *trixi*:
```
pip install trixi
```


Or to always get the newest version you can install *trixi* directly via git:
```
git clone https://github.com/MIC-DKFZ/trixi.git
cd trixi
pip install -e .
```

# Documentation

The docs can be found here: [trixi.rtfd.io](https://trixi.readthedocs.io/en/latest/)

Or you can build your own docs using Sphinx.

#### Sphinx Setup

Install Sphinx (fixed to 1.7.0 for now because of issues with Readthedocs):  
`pip install sphinx==1.7.0`

Generate HTML:  
`path/to/PROJECT/doc$ make html`

index.html will be at:  
`path/to/PROJECT/doc/_build/html/index.html`

#### Notes
* Rerun `make html` each time existing modules are updated (this will automatically call sphinx-apidoc)
* Do not forget indent or blank lines
* Code with no classes or functions is not automatically captured using apidoc


#### Example Documentation

We use Google style docstrings:

	def show_image(self, image, name, file_format=".png", **kwargs):
        """
        This function shows an image.

        Args:
            image(np.ndarray): image to be shown
            name(str): image title
        """


# Examples

Examples can be found here for:
* [Visdom-Logger](https://github.com/MIC-DKFZ/trixi/blob/master/examples/numpy_visdom_logger_example.ipynb)
* [Experiment-Logger](https://github.com/MIC-DKFZ/trixi/blob/master/examples/pytorch_example.ipynb)
* [Experiment Infrastructure](https://github.com/MIC-DKFZ/trixi/blob/master/examples/pytorch_experiment.ipynb)
</b>(with a
 simple MNIST Experiment example and resuming and comparison of different hyperparameters)
