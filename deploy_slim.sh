#!/bin/sh
sed -i "s/requirements_full.txt/requirements.txt/g" setup.py
sed -i "s/name='trixi'/name='trixi-slim'/g" setup.py
python setup.py sdist bdist_wheel
twine upload dist/* -u $PYPI_USER -p $PYPI_PASSWORD
sed -i "s/requirements.txt/requirements_full.txt/g" setup.py
sed -i "s/name='trixi-slim'/name='trixi'/g" setup.py
