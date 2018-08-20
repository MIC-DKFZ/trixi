[![DOI](https://zenodo.org/badge/134823632.svg)](https://zenodo.org/badge/latestdoi/134823632)

# ![logo_small](https://github.com/MIC-DKFZ/trixi/blob/master/trixi/doc/_static/logo/trixi-small.png)

Manage your machine learning experiments.  
![icon](https://github.com/MIC-DKFZ/trixi/blob/master/trixi/experiment_browser/static/assets/trixi-icon.png)
trixi is a tool to help you configure, visualize and log your experiments in a reproducible fashion.    

* [Features](#features)
* [Installation](#installation)
* [Sphinx Setup](#sphinx-setup)
* [Remote Usage](#remote-usage)

## Features
### Visdom
![icon](https://github.com/MIC-DKFZ/trixi/blob/master/trixi/experiment_browser/static/assets/trixi-icon.png)
trixi integrates with [Visdom](https://github.com/facebookresearch/visdom) and offers the complete Visdom functionality.
![visdom](https://lh3.googleusercontent.com/-h3HuvbU2V0SfgqgXGiK3LPghE5vqvS0pzpObS0YgG_LABMFk62JCa3KVu_2NV_4LJKaAa5-tg=s0)

### trixi Experiment Browser
![icon](https://github.com/MIC-DKFZ/trixi/blob/master/trixi/experiment_browser/static/assets/trixi-icon.png)
trixi's experimt browser offers a complete overview of experiments along with all config parameters, an interactive
comparison highlighting differences in the configs and a detailed view of all images, plots, results and logs of each experiment.
![trixi browser](https://github.com/MIC-DKFZ/trixi/blob/master/doc/_static/trixi_browser.gif)

## Installation
Install dependencies:
```
pip install -r requirements.txt
```

If you want to use the full functionallity e.g. any of the PyTorch loggers or the Experiment class:
```
pip install -r requirements_full.txt
```

Install trixi:
```
git clone https://github.com/MIC-DKFZ/trixi.git
cd trixi
pip install -e .
```

## Sphinx Setup

### Setup

Install Sphinx:
`pip install sphinx`

Generate Api-docs:
`path/to/PROJECT/doc$ sphinx-apidoc -f -o . ..`

Open index.html:
`firefox path/to/PROJECT/doc/_build/html/index.html`

### Notes
* rerun make html each time existing modules are updated
* DO NOT forget indent or blank lines
* Code with no classes or functions is not automatically captured using apidoc


### Example Documentation

This follows the Google style docstring guidelines:

	def show_image(self, image, name, file_format=".png", **kwargs):
        """
        This function shows an image.

        Args:
            image(np.ndarray): image to be shown
            name(str): image title
        """


**IMPORTANT NOTE**: Somehow pytorch and lasagne/theano do not play nicely together. So if you
import lasagne/theano and trixi (which imports pytorch if you have it installed),
your program will get stuck. So you can only use trixi with lasagne/theano if you do not
have pytorch installed. If you need both you can use virtual_envs.

## Remote Usage

### Use on remote server in same network
Simple run visdom on remote server and then on your local computer go to `MY_REMOTE_SERVER_NAME:8080`.

### Use on remote server in different network

If you want to run trixi on a remote server, but show the results locally
you can do:

```
# On local computer:
ssh -N -f -L localhost:8080:localhost:8080 USERNAME@REMOTE_SERVERNAME

# On remote server:
python -m visdom.server -port 8080
python my_random_trixi_script.py
```

Now on your local computer you can go to `localhost:8080` and see the visdom dashboard.
* [API](#api)
* [To Do](#to-do)
* [Contributing](#contributing)
