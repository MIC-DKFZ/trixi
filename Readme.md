[![DOI](https://zenodo.org/badge/134823632.svg)](https://zenodo.org/badge/latestdoi/134823632)
[![Build Status](https://travis-ci.org/MIC-DKFZ/trixi.svg?branch=master)](https://travis-ci.org/MIC-DKFZ/trixi)

# ![logo_small](https://github.com/MIC-DKFZ/trixi/blob/master/doc/_static/logo/trixi-small.png)

Manage your machine learning experiments.  
![icon](https://github.com/MIC-DKFZ/trixi/blob/master/trixi/experiment_browser/static/assets/trixi-icon.png)
trixi is a tool to help you configure, visualize and log your experiments in a reproducible fashion.    

* [Features](#features)
* [Installation](#installation)
* [Docs](#docs)

## Features

Trixi consists of three parts:
* Logging API ( e.g. local, visdom)
* Experiment Infrastructure ( for reproducibility with easy config and checkpointing)
* Experiment Browser (for evaluation of your experiments)

An overview is given [here](https://trixi.readthedocs.io/en/latest/class_diagram.html).

### Logging API

The Logging-Api provides a standardized way for logging results to different backends. 
The logging api supports 
(among others):
* Values
* Text
* Plots (Bar, Line, Scatter, Piechart, ...)
* Images (Single, as grid)

And offers different Backends, e.g. :
* Visdom ([visdom-loggers](https://trixi.readthedocs.io/en/latest/api/trixi.logger.visdom.html))
* Matplotlib / Seaborn ([plt-loggers](https://trixi.readthedocs.io/en/latest/api/trixi.logger.plt.html))
* Local Disk ([file-loggers](https://trixi.readthedocs.io/en/latest/api/trixi.logger.file.html))
* Telegram ([message-loggers](https://trixi.readthedocs.io/en/latest/api/trixi.logger.message.html))

And an [Experiment-logger](https://trixi.readthedocs.io/en/latest/api/trixi.logger.experiment.experimentlogger.html) for logging your experiments, which automatically creates a structured directory and allows 
storing of config, results, plots, dict, array, images, ...

Here are some examples:

* [Visdom](https://github.com/facebookresearch/visdom):
![visdom](https://lh3.googleusercontent.com/-h3HuvbU2V0SfgqgXGiK3LPghE5vqvS0pzpObS0YgG_LABMFk62JCa3KVu_2NV_4LJKaAa5-tg=s0)

### Experiment Infrastructure

The [Experiment Infrastructure](https://trixi.readthedocs.io/en/latest/api/trixi.experiment.pytorchexperiment.html) provides a unified way to configure, run, store and evaluate your results.
It provides you an Experiment-Interface, for which you can implement the training, validation and testing.
Furthermore it automatically provides you with easy access to the Logging API and stores your config es well as the 
results for easy evaluation and reproduction of different experiments.

For more info, visit the Documentation.

### Experiment Browser
The Experiment Browser offers a complete overview of experiments along with all config parameters and results.
It also allows to combine and/or compare different experiments. 
It also gives an interactive comparison highlighting differences in the configs and a detailed view of all images, 
plots, results and logs of each experiment, with live plots and more.
![trixi browser](https://github.com/MIC-DKFZ/trixi/blob/master/doc/_static/trixi_browser.gif)

## Installation
Install dependencies:
```
pip install trixi
```


Or to always get the newest version you can install trixi directly via git:
```
git clone https://github.com/MIC-DKFZ/trixi.git
cd trixi
pip install -e .
```

## Docs

The docs can be found here: [trixi.rtfd.io](https://trixi.readthedocs.io/en/latest/)

Or you can build your own docs using Sphinx.

### Sphinx Setup

#### Setup

Install Sphinx:
`pip install sphinx`

Generate Api-docs:
`path/to/PROJECT/doc$ sphinx-apidoc -f -o . ..`

Open index.html:
`firefox path/to/PROJECT/doc/_build/html/index.html`

#### Notes
* rerun make html each time existing modules are updated
* DO NOT forget indent or blank lines
* Code with no classes or functions is not automatically captured using apidoc


#### Example Documentation

This follows the Google style docstring guidelines:

	def show_image(self, image, name, file_format=".png", **kwargs):
        """
        This function shows an image.

        Args:
            image(np.ndarray): image to be shown
            name(str): image title
        """


* [API](#api)
