init:
	pip install numpy ; \
  pip install -f http://www.simpleitk.org/SimpleITK/resources/software.html --trusted-host www.simpleitk.org -r requirements.txt

tests:
	python -m unittest discover

install_develop:
	python setup.py develop

install:
	python setup.py install

documentation:
	sphinx-apidoc -e -f susi -o doc/
