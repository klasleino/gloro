import numpy as np

from abc import abstractmethod
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.callbacks import LearningRateScheduler
from time import time

from gloro.training.losses import Trades
from gloro.training.schedules import Schedule
from gloro.utils import get_value
from gloro.utils import print_if_verbose
from gloro.utils import set_value


class VariableScheduler(Callback):

    def __init__(
        self, 
        variable_name,
        schedule_string='fixed', 
        test_value=None,
        base_value=None, 
        interpolate=True,
        verbose=True,
    ):
        super().__init__()

        self._name = variable_name

        self._schedule_string = schedule_string
        self._interpolate_between_epochs = interpolate

        self._base_value = base_value
        self._test_value = test_value

        self._print = print_if_verbose(verbose)

    @abstractmethod
    def get_var(self):
        raise NotImplementedError

    @abstractmethod
    def set_var(self, value):
        raise NotImplementedError

    def on_train_begin(self, logs=None):

        self._num_steps = self.params['steps']

        self._schedule = Schedule(
            self._schedule_string, 
            duration=self.params['epochs'], 
            base_value=(
                self._base_value if self._base_value is not None else
                self.get_var()))

        if self._test_value is None:
            self._test_value = self.get_var()

        self.__prev = None
        self.__current = self._schedule[0]

    def on_test_begin(self, logs=None):
        self._train_value = self.get_var()
        self.set_var(self._test_value)

    def on_test_end(self, logs=None):
        self.set_var(self._train_value)

    def on_epoch_begin(self, epoch, logs=None):
        if epoch < len(self._schedule):
            next_value = self._schedule[epoch]

            self.set_var(next_value)
        
            if self.__prev is None or self.__prev != self.__current:
                self._print(f'---- setting {self._name}={next_value:.5f} ----')

            self.__prev = self.__current
            self.__current = next_value

    def on_train_batch_begin(self, batch, logs=None):
        if self._interpolate_between_epochs:
            self.set_var(
                self.__prev + (self.__current - self.__prev) * (
                    batch / self._num_steps))

    def on_train_end(self, logs=None):
        self.set_var(self._test_value)


class EpsilonScheduler(VariableScheduler):

    def __init__(
        self,
        schedule_string='fixed', 
        test_value=None,
        base_value=None, 
        interpolate=True,
        verbose=True,
    ):
        super().__init__(
            'epsilon', 
            schedule_string=schedule_string,
            test_value=test_value,
            base_value=base_value,
            interpolate=interpolate,
            verbose=verbose)

    def get_var(self):
        return self.model.epsilon

    def set_var(self, value):
        self.model.epsilon = value


class TradesScheduler(VariableScheduler):

    def __init__(
        self,
        schedule_string='fixed', 
        test_value=None,
        base_value=None, 
        interpolate=True,
        verbose=True,
    ):
        super().__init__(
            'TRADES parameter', 
            schedule_string=schedule_string,
            test_value=test_value,
            base_value=base_value,
            interpolate=interpolate,
            verbose=verbose)

    def get_var(self):
        if isinstance(self.model.loss, dict):
            loss = self.model.loss['pred']
        else:
            loss = self.model.loss

        # Make sure that the model is actually using TRADES loss.
        if not isinstance(loss, Trades):
            raise ValueError(
                '`TradesScheduler` can only be used with `Trades` loss.')

        return loss.lam

    def set_var(self, value):
        if isinstance(self.model.loss, dict):
            loss = self.model.loss['pred']
        else:
            loss = self.model.loss

        # Make sure that the model is actually using TRADES loss.
        if not isinstance(loss, Trades):
            raise ValueError(
                '`TradesScheduler` can only be used with `Trades` loss.')

        loss.lam = value


class LrScheduler(VariableScheduler):

    def __init__(
        self,
        schedule_string='fixed', 
        test_value=None,
        base_value=None, 
        interpolate=True,
        verbose=True,
    ):
        super().__init__(
            'learning rate', 
            schedule_string=schedule_string,
            test_value=test_value,
            base_value=base_value,
            interpolate=interpolate,
            verbose=verbose)

    def get_var(self):
        return get_value(self.model.optimizer.learning_rate)

    def set_var(self, value):
        set_value(self.model.optimizer.learning_rate, value)


class UpdatePowerIterates(Callback):
    def __init__(
        self, 
        convergence_threshold=1e-4,
        short_convergence_threshold=1e-2,
        iteration_batch_size=100, 
        do_initial_convergence=True,
        verbose=True,
    ):
        super().__init__()

        self._convergence_threshold = convergence_threshold
        self._short_convergence_threshold = short_convergence_threshold
        self._batch_size = iteration_batch_size
        self._do_initial_convergence = do_initial_convergence
        self._verbose = verbose

        self._print = print_if_verbose(verbose)

        self._converged = False

    def on_train_begin(self, logs=None):

        if self._do_initial_convergence:

            self._print('---- refreshing iterates ----')

            start = time()

            self.model.refresh_iterates(
                convergence_threshold=self._short_convergence_threshold,
                batch_size=self._batch_size,
                verbose=self._verbose)

            self._print(f'   > done: {(time() - start):.2f} seconds')

    def on_test_begin(self, logs=None):
        self._print('\n---- refreshing iterates ----')

        start = time()

        self.model.refresh_iterates(
            convergence_threshold=self._short_convergence_threshold,
            batch_size=self._batch_size,
            verbose=self._verbose)

        self._print(f'   > done: {(time() - start):.2f} seconds')

        self._converged = True

    def on_epoch_begin(self, epoch, logs=None):
        self._converged = False
    
    def on_epoch_end(self, epoch, logs=None):
        if not self._converged:
            self._print('\n---- refreshing iterates ----')

            start = time()

            self.model.refresh_iterates(
                convergence_threshold=self._short_convergence_threshold,
                batch_size=self._batch_size,
                verbose=self._verbose)

            self._print(f'   > done: {(time() - start):.2f} seconds')

    def on_train_end(self, logs=None):
        self._print('---- refreshing iterates precisely ----')

        start = time()
        self.model.refresh_iterates(
            convergence_threshold=self._convergence_threshold,
            batch_size=self._batch_size,
            verbose=self._verbose)

        self._print(f'   > done: {(time() - start):.2f} seconds')
