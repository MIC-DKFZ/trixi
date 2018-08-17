init:
	pip install numpy ; \
  pip install -f http://www.simpleitk.org/SimpleITK/resources/software.html --trusted-host www.simpleitk.org -r requirements.txt


tests:
	python -m unittest discover


install:
	python setup.py install


install_develop:
	python setup.py develop


documentation:
	sphinx-apidoc -e -f trixi -o doc/api
