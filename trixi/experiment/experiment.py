import time


class Experiment(object):
    """
    An abstract Experiment which can be run for a number of epochs.

    The basic life cycle of an experiment is::

        setup()
        prepare()

        while epoch < n_epochs:
            train()
            validate()
            epoch += 1

        end()

    If you want to use another criterion than number of epochs, e.g. stopping based on validation loss,
    you can implement that in your validation method and just call .stop() at some point to break the loop.
    Just set your n_epochs to a high number or np.inf.

    The reason there is both :meth:`.setup` and :meth:`.prepare` is that internally there is also
    a :meth:`._setup_internal` method for hidden magic in classes that inherit from this. For
    example, the :class:`trixi.experiment.pytorchexperiment.PytorchExperiment` uses this to restore checkpoints. Think
    of :meth:`.setup` as an :meth:`.__init__` that is only called when the Experiment is actually
    asked to do anything. Then use :meth:`.prepare` to modify the fully instantiated Experiment if
    you need to.

    To write a new Experiment simply inherit the Experiment class and overwrite the methods.
    You can then start your Experiment calling :meth:`.run`

    In Addition the Experiment also has a test function. If you call the :meth:`.run_test` method it
    will call the :meth:`.test` and :meth:`.end_test` method internally (and if you give the
    parameter setup = True in run_test is will again call :meth:`.setup` and :meth:`.prepare` ).

    Each Experiment also has its current state in  :attr:`_exp_state`, its start time in
    :attr:`_time_start`, its end time in :attr:`_time_end` and the current epoch index in
    :attr:`_epoch_idx`

    Args:
        n_epochs (int): The number of epochs in the Experiment (how often the train and validate
            method will be called)

    """

    def __init__(self, n_epochs=0):

        self.n_epochs = n_epochs
        self._exp_state = "Preparing"
        self._time_start = ""
        self._time_end = ""
        self._epoch_idx = 0
        self.__stop = False

    def run(self, setup=True):
        """
        This method runs the Experiment. It runs through the basic lifecycle of an Experiment::

            setup()
            prepare()

            while loop(epoch, train_result, validate_result):
                train()
                validate()
                end_epoch(train_result, validate_result)
                epoch += 1

            end()

        The loop function decides whether to continue, depending on the epoch and the results from the previous epoch.
        In the default implementation, it just compares the current epoch to the specified number of epochs.

        """

        try:
            self._time_start = time.strftime("%y-%m-%d_%H:%M:%S", time.localtime(time.time()))
            self._time_end = ""

            if setup:
                self.setup()
                self._setup_internal()

            self.prepare()

            self._exp_state = "Started"
            self._start_internal()
            print("Experiment started.")

            self.__stop = False
            self._train_result = None
            self._validate_result = None
            while self.loop(epoch=self._epoch_idx, train_result=self._train_result, validate_result=self._validate_result) and not self.__stop:
                self._train_result = self.train(epoch=self._epoch_idx)
                self._validate_result = self.validate(epoch=self._epoch_idx)
                self.end_epoch(train_result=self._train_result, validate_result=self._validate_result)
                self._end_epoch_internal(epoch=self._epoch_idx)
                self._epoch_idx += 1

            self._exp_state = "Trained"
            print("Training complete.")

            self.end()
            self._end_internal()
            self._exp_state = "Ended"
            print("Experiment ended.")

            self._time_end = time.strftime("%y-%m-%d_%H:%M:%S", time.localtime(time.time()))

        except Exception as e:

            self._exp_state = "Error"
            self._time_end = time.strftime("%y-%m-%d_%H:%M:%S", time.localtime(time.time()))
            self.process_err(e)

    def run_test(self, setup=True):
        """
        This method runs the Experiment.

        The test consist of an optional setup and then calls the :meth:`.test` and :meth:`.end_test`.

        Args:
            setup: If True it will execute the :meth:`.setup` and :meth:`.prepare` function similar to the run method
                before calling :meth:`.test`.

        """

        try:

            if setup:
                self.setup()
                self._setup_internal()

            self.prepare()

            self._exp_state = "Testing"
            print("Start test.")

            self.test()

            self.end_test()
            self._end_test_internal()

            self._exp_state = "Tested"
            print("Testing complete.")

        except Exception as e:

            self._exp_state = "Error"
            self.process_err(e)

    def loop(self, epoch, train_result=None, validate_result=None):
        """This method decides whether the train loop will continue."""

        if epoch < self.n_epochs:
            return True
        else:
            return False

    @property
    def epoch(self):
        """Convenience access property for self._epoch_idx"""
        return self._epoch_idx

    def setup(self):
        """Is called at the beginning of each Experiment run to setup the basic components needed for a run."""
        pass

    def train(self, epoch):
        """
        The training part of the Experiment, it is called once for each epoch

        Args:
            epoch (int): The current epoch the train method is called in

        """
        pass

    def validate(self, epoch):
        """
        The evaluation/validation part of the Experiment, it is called once for each epoch (after the training
        part)

        Args:
            epoch (int):  The current epoch the validate method is called in

        """
        pass

    def test(self):
        """The testing part of the Experiment"""
        pass

    def end_epoch(self, train_result=None, validate_result=None):
        """Do anything you need to with the result from train and validate in the current epoch."""
        pass

    def stop(self):
        """If called the Experiment will stop after that epoch and not continue training"""
        self.__stop = True

    def process_err(self, e):
        """
        This method is called if an error occurs during the execution of an experiment. Will just raise by default.

        Args:
            e (Exception): The exception which was raised during the experiment life cycle

        """
        raise e

    def _setup_internal(self):
        pass

    def _end_epoch_internal(self, epoch):
        pass

    def end(self):
        """Is called at the end of each experiment"""
        pass

    def end_test(self):
        """Is called at the end of each experiment test"""
        pass

    def prepare(self):
        """This method is called directly before the experiment training starts"""
        pass

    def _start_internal(self):
        pass

    def _end_internal(self):
        pass

    def _end_test_internal(self):
        pass
