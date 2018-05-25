# Install
Install Dependencies
```
pip install numpy seaborn matplotlib
```

If you want to use any of the PyTorch loggers or the Experiment class:
```
#install pytorch: http://pytorch.org/
pip install torchvision
```

Install trixi:
```
git clone https://phabricator.mitk.org/source/trixi.git
cd trixi
pip install -e .
```

# Sphinx Setup

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

# Use on remote server in same network
Simple run visdom on remote server and then on your local computer go to `MY_REMOTE_SERVER_NAME:8080`.

# Use on remote server in different network

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
