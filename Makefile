tests:
	python -m unittest discover


install:
	python setup.py install


install_develop:
	python setup.py develop


documentation:
	sphinx-apidoc -e -f trixi -o doc/api
