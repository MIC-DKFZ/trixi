Sphinx Setup
=================


Setup
-----
#. Install Sphinx::

	pip install sphinx

#. Api-doc Generation::

	path/to/PROJECT/doc$ sphinx-apidoc -f -o . ..

#. Make html::

	path/to/PROJECT/doc$ make html

#. Open index.html::
	
	firefox path/to/PROJECT/doc/_build/html/index.html

Notes
-----
* rerun apidoc each time you added new modules
* rerun make html each time existing modules are updated
* DO NOT forget indent or blank lines
* Code with no classes or functions is not automatically captured using apidoc



Example Documentation
---------------------
This follows the Google style docstring guidelines:
:: 

	def show_image(self, image, name, file_format=".png", **kwargs):
        """
        This function shows an image.

        Args:
            image(np.ndarray): image to be shown
            name(str): image title
        """

For rendered version see :meth:`vislogger.experimentlogger.ExperimentLogger.show_image`

.. inheritance-diagram:: vislogger.abstractlogger.AbstractLogger vislogger.experimentlogger.ExperimentLogger
	:parts: 1 


