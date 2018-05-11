install:
	python setup.py install

documentation:
	sphinx-apidoc -e -f vislogger -o doc/
