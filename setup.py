import os
import re

from setuptools import setup, find_packages


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


def read_file(file):
    with open(file) as f:
        content = f.read()
    return content


def find_version(file):
    content = read_file(file)
    version_match = re.search(r"^__version__ = ['\"]([^'\"]*)['\"]", content, re.M)
    if version_match:
        return version_match.group(1)
    raise RuntimeError("Unable to find version string.")


required = resolve_requirements(os.path.join(os.path.dirname(__file__), 'requirements_full.txt'))
readme = read_file(os.path.join(os.path.dirname(__file__), "Readme.md"))
#license = read_file(os.path.join(os.path.dirname(__file__), "LICENSE"))
version = find_version(os.path.join(os.path.dirname(__file__), "trixi", "__init__.py"))


setup(name='trixi',
      version=version,
      description='Manage your machine learning experiments with trixi - modular, reproducible, high fashion',
      long_description=readme,
      long_description_content_type="text/markdown",
      url='https://github.com/MIC-DKFZ/trixi',
      author='Medical Image Computing Group, DKFZ',
      author_email='mic@dkfz-heidelberg.de',
      license="MIT",
      packages=find_packages(),
      install_requires=required,
      zip_safe=True,
      entry_points={
          'console_scripts': ['trixi-browser=trixi.experiment_browser.browser:start_browser'],
      },
      include_package_data=True
      )
