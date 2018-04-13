Sphinx Setup
=================


Setup
-----
#. Install Sphinx::

	pip install sphinx

#. Api-doc Generation::

	path/to/PROJECT/docs/source$ sphinx-apidoc -f -o . ../..

#. Make html::

	path/to/PROJECT/docs$ make html

#. Open index.html::
	
	firefox path/to/PROJECT/docs/build/html/index.html

Notes
-----
* rerun apidoc each time you added new modules
* rerun make html each time existing modules are updated
* DO NOT forget indent or blank lines
* Code with no classes or functions is not automatically captured using apidoc



Example Documentation
---------------------
This follows the Sphinx docstring guidelines (not Google):
:: 

	def show_image(self, image, name, file_format=".png", **kwargs):
		"""
		This function shows an image.

		:param image: image to be shown
		:type image: np.ndarray
		:param name: image title
		:type name: str
		"""

For rendered version see :meth:`vislogger.experimentlogger.ExperimentLogger.show_image`


