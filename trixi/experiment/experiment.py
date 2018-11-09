import time


class Experiment(object):
    """
    An abstract Experiment which can be run for a number of epochs.

    The basic life cycle of an experiment is::

        setup()
        prepare()

        for epoch in n_epochs:
            train()
            validate()

        end()

    To write a new Experiment simply inherit the Experiment class and overwrite the methods.
    You can then start your experiment calling :meth:`.run`

    In Addition the Experiment also has a test function. If you call the :meth:`.run_test` method is will call the
    :meth:`.test` and :meth:`.end_test` method internally (and if you give the parameter setup = True
    in run_test is will again call :meth:`.setup` and :meth:`.prepare` ).

    Each experiment also has its current state in  :attr:`_exp_state`, its start time in  :attr:`_time_start`,
    its end time in :attr:`_time_end` and the current epoch index in :attr:`_epoch_idx`

    """


    def __init__(self, n_epochs=0):
        """
        Initializes a new Experiment with a given number of epochs

        Args:
            n_epochs (int): The number of epochs in the experiment (how often the train and validate method
                will be called)
        """

        self.n_epochs = n_epochs
        self._exp_state = "Preparing"
        self._time_start = ""
        self._time_end = ""
        self._epoch_idx = 0

    def run(self):
        """
        This method runs the experiment. It runs through the basic lifecycle of an experiment::

            setup()
            prepare()

            for epoch in n_epochs:
                train()
                validate()

            end()

        """

        try:
            self._time_start = time.strftime("%y-%m-%d_%H:%M:%S", time.localtime(time.time()))
            self._time_end = ""

            self.setup()
            self._setup_internal()
            self.prepare()

            self._exp_state = "Started"
            self._start_internal()
            print("Experiment started.")

            for epoch in range(self._epoch_idx, self.n_epochs):
                self.train(epoch=epoch)
                self.validate(epoch=epoch)
                self._end_epoch_internal(epoch=epoch)
                self._epoch_idx += 1

            self._exp_state = "Trained"
            print("Training complete.")

            self.end()
            self._end_internal()
            self._exp_state = "Ended"
            print("Experiment ended.")

            self._time_end = time.strftime("%y-%m-%d_%H:%M:%S", time.localtime(time.time()))

        except Exception as e:

            # run_error = e
            # run_error_traceback = traceback.format_tb(e.__traceback__)
            self._exp_state = "Error"
            self.process_err(e)
            self._time_end = time.strftime("%y-%m-%d_%H:%M:%S", time.localtime(time.time()))

            raise e

    def run_test(self, setup=True):
        """
        This method runs the experiment.

        The test consist of an optional setup and then calls the :meth:`.test` and :meth:`.end_test` .


        Args:
            setup: If True it will execute the :meth:`.setup` and :meth:`.prepare` function similar to the run method
            before calling :meth:`.test` .

        """

        """"""

        try:

            if setup:
                self.setup()
                self._setup_internal()
                self.prepare()

            self._exp_state = "Testing"
            print("Start test.")

            self.test()
            self.end_test()
            self._exp_state = "Tested"

            self._end_test_internal()

            print("Testing complete.")

        except Exception as e:

            # run_error = e
            # run_error_traceback = traceback.format_tb(e.__traceback__)
            self._exp_state = "Error"
            self.process_err(e)

            raise e

    def setup(self):
        """Is called at the beginning of each experiment run to setup the basic components needed for a run"""
        pass

    def train(self, epoch):
        """
        The training part of the experiment, it is called once for each epoch

        Args:
            epoch (int): The current epoch the train method is called in

        """
        pass

    def validate(self, epoch):
        """
        The evaluation/validation part of the experiment, it is called once for each epoch (after the training
        part)

        Args:
            epoch (int):  The current epoch the validate method is called in

        """
        pass

    def test(self):
        """The testing part of the experiment"""
        pass

    def process_err(self, e):
        """
        This method is called if an error occurs during the execution of an experiment

        Args:
            e (Exception): The exception which was raised during the experiment life cycle

        """
        pass

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
