Class Diagram
=================

Logger
---------
.. inheritance-diagram:: trixi.logger.experiment.pytorchexperimentlogger trixi.logger.visdom.pytorchvisdomlogger trixi.logger.message.telegrammessagelogger trixi.logger.file.textfilelogger trixi.logger.file.pytorchplotfilelogger
   :top-classes: trixi.logger.abstractlogger
   :parts: 1

Experiment
------------
.. inheritance-diagram:: trixi.experiment.pytorchexperiment
   :top-classes: trixi.experiment.experiment
   :parts: 1
