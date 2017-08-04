from setuptools import setup

setup(name='vislogger',
      version='0.1',
      description='Package to log visualizations, e.g. with visdom',
      url='http://phabricator.mitk.org/source/vislogger',
      author='Medical Image Computing Group, DKFZ',
      author_email='mic@dkfz-heidelberg.de',
      license='MIT',
      packages=['vislogger'],
      install_requires=['numpy', 'visdom'],
      zip_safe=True)
