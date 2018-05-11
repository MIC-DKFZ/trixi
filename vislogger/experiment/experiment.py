import time


class Experiment(object):
    def __init__(self, n_epochs=0):
        # super(Experiment, self).__init__()

        self.n_epochs = n_epochs
        self.exp_state = "Preparing"
        self.time_start = ""
        self.time_end = ""
        self.epoch_idx = 0

    def run(self):
        """This method runs the experiment"""

        try:
            self.time_start = time.strftime("%y-%m-%d_%H:%M:%S", time.localtime(time.time()))
            self.time_end = ""

            self.setup()
            self._setup_internal()
            self.prepare()

            self.exp_state = "Started"
            self._start_internal()
            print("Experiment started.")

            for epoch in range(self.n_epochs):
                self.epoch_idx = epoch
                self.train(epoch=epoch)
                self.validate(epoch=epoch)
                self._end_epoch_internal(epoch=epoch)

            self.exp_state = "Trained"
            print("Training complete.")

            self.end()
            self.exp_state = "Ended"
            print("Experiment ended.")

            self.time_end = time.strftime("%y-%m-%d_%H:%M:%S", time.localtime(time.time()))

        except Exception as e:

            # run_error = e
            # run_error_traceback = traceback.format_tb(e.__traceback__)
            self.exp_state = "Error"
            self.process_err(e)
            self.time_end = time.strftime("%y-%m-%d_%H:%M:%S", time.localtime(time.time()))

            raise e

    def run_test(self, setup=True):
        """This method runs the experiment"""

        try:

            if setup:
                self.setup()
                self._setup_internal()
                self.prepare()

            self.exp_state = "Testing"
            print("Start test.")

            self.test()
            self.end_test()

            self.exp_state = "Tested"
            print("Testing complete.")

        except Exception as e:

            # run_error = e
            # run_error_traceback = traceback.format_tb(e.__traceback__)
            self.exp_state = "Error"
            self.process_err(e)

            raise e

    def setup(self):
        """Is called at the beginning of each experiment run to setup the basic components needed for a run"""
        pass

    def train(self, epoch):
        """The training part of the experiment, it is called once for each epoch"""
        pass

    def validate(self, epoch):
        """The evaluation/valdiation part of the experiment, it is called once for each epoch (after the training
        part)"""
        pass

    def test(self):
        """The testing part of the experiment"""
        pass

    def process_err(self, e):
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
