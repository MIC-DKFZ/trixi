from setuptools import setup

with open("README.md") as f:
    readme = f.read()

with open("LICENSE") as f:
    license = f.read()

setup(name='vislogger',
      version='0.1',
      description='Package to log visualizations, e.g. with visdom',
      long_description=readme,
      url='http://phabricator.mitk.org/source/vislogger',
      author='Medical Image Computing Group, DKFZ',
      author_email='mic@dkfz-heidelberg.de',
      license=license,
      packages=['vislogger'],
      install_requires=['numpy', 'visdom', 'graphviz', 'matplotlib', 'seaborn', 'portalocker'],
      zip_safe=True)
