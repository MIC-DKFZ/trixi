Installation
============
 Install Dependencies:: 

	pip install numpy seaborn matplotlib


If you want to use pytorch*logger:

Install pytorch: http://pytorch.org/
	
:: 

	pip install torchvision


 Install vislogger:: 

	git clone https://phabricator.mitk.org/source/vislogger.git
	cd vislogger
	pip install -e .


**IMPORTANT NOTE**: Somehow pytorch and lasagne/theano do not play nicely together. So if you 
import lasagne/theano and vislogger (which imports pytorch if you have it installed), 
your program will get stuck. So you can only use vislogger with lasagne/theano if you do not 
have pytorch installed. If you need both you can use virtual_envs.

* Use on remote server in same network
Simple run visdom on remote server and then on your local computer go to `MY_REMOTE_SERVER_NAME:8080`.

* Use on remote server in different network
If you want to run vislogger on a remote server, but show the results locally
you can do:


* On local computer::

	ssh -N -f -L localhost:8080:localhost:8080 USERNAME@REMOTE_SERVERNAME

* On remote server::

	python -m visdom.server -port 8080
	python my_random_vislogger_script.py

Now on your local computer you can go to `localhost:8080` and see the visdom dashboard.
