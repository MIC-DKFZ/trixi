from setuptools import setup
import os

with open(os.path.join(os.path.dirname(__file__), "Readme.md")) as f:
    readme = f.read()

with open(os.path.join(os.path.dirname(__file__), "LICENSE")) as f:
    license = f.read()

setup(name='trixi',
      version='0.1',
      description='Package to log visualizations and experiments, e.g. with visdom',
      long_description=readme,
      url='http://phabricator.mitk.org/source/vislogger',
      author='Medical Image Computing Group, DKFZ',
      author_email='mic@dkfz-heidelberg.de',
      license=license,
      packages=['trixi'],
      install_requires=['numpy', 'visdom', 'graphviz', 'matplotlib', 'seaborn', 'portalocker'],
      zip_safe=True)
