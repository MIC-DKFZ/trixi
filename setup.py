import os

from setuptools import setup


def resolve_requirements(file):
    requirements = []
    with open(file) as f:
        req = f.read().splitlines()
        for r in req:
            if r.startswith("-r"):
                requirements += resolve_requirements(os.path.join(os.path.dirname(file), r.split(" ")[1]))
            else:
                requirements.append(r)
    return requirements


with open(os.path.join(os.path.dirname(__file__), "Readme.md")) as f:
    readme = f.read()

with open(os.path.join(os.path.dirname(__file__), "LICENSE")) as f:
    license = f.read()

required = resolve_requirements(os.path.join(os.path.dirname(__file__), 'requirements_full.txt'))

setup(name='trixi',
      version='0.1.0.1',
      description='Package to log visualizations and experiments, e.g. with visdom',
      long_description=readme,
      url='https://github.com/MIC-DKFZ/trixi',
      author='Medical Image Computing Group, DKFZ',
      author_email='mic@dkfz-heidelberg.de',
      license=license,
      packages=['trixi'],
      install_requires=required,
      zip_safe=True,
      entry_points={
          'console_scripts': ['trixi-browser=trixi.experiment_browser.browser:start_browser'],
      }
      )
