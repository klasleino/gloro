import numpy as np

from tensorflow.keras.callbacks import Callback
from tensorflow.keras.callbacks import LearningRateScheduler

import gloro

from gloro.training.losses import Trades


def _print_if_verbose(verbose):
    if verbose:
        return lambda s: print(s)

    else:
        return lambda s: None


def _check_is_gloro_net(callback, model):
    if not isinstance(model, gloro.GloroNet):
        raise ValueError(
            f'`{callback.__class__.__name__}` can only be used with a '
            f'`GloroNet` model')


def _parse_schedule_string(schedule_string, duration):
    if schedule_string == 'fixed':
        return [1.]

    elif schedule_string == 'linear_half':
        return [
            i / (duration // 2) for i in range(duration // 2 + 1)
        ]

    elif schedule_string == 'linear':
        return [
            i / (duration - 1) for i in range(duration)
        ]

    elif schedule_string == 'logarithmic':
        initial = 0.01
        final = 1.

        return [
            np.log(
                (np.exp(final) - np.exp(initial)) / (duration - 1) * i +
                np.exp(initial))
            for i in range(duration)
        ]

    elif schedule_string == 'logarithmic_half':
        initial = 0.01
        final = 1.

        return [
             np.log(
                (np.exp(final) - np.exp(initial)) / (duration // 2) * i +
                np.exp(initial))
            for i in range(duration // 2 + 1)
        ]

    else:
        raise ValueError(f'unrecognized schedule string: {schedule_string}')


class EpsilonScheduler(Callback):

    def __init__(self, schedule_string='fixed', verbose=True):
        super().__init__()

        self._schedule_string = schedule_string
        self._print = _print_if_verbose(verbose)

    def on_train_begin(self, logs=None):
        _check_is_gloro_net(self, self.model)

        schedule = np.array(_parse_schedule_string(
            self._schedule_string, self.params['epochs']))

        # We assume the epsilon originally set on the model is the final epsilon
        # we will train with, and scale the schedule accordingly.
        self._epoch_to_eps = self.model.epsilon * schedule

        self.__prev_eps = None
        self.__current_eps = self._epoch_to_eps[0]

    def on_epoch_begin(self, epoch, logs=None):
        if epoch < len(self._epoch_to_eps):
            next_eps = self._epoch_to_eps[epoch]

            self.model.epsilon = next_eps
        
            if self.__prev_eps is None or self.__prev_eps != self.__current_eps:
                self._print(f'---- setting epsilon={next_eps:.3f} ----')

            self.__prev_eps = self.__current_eps
            self.__current_eps = next_eps


class TradesScheduler(Callback):

    def __init__(self, schedule_string='fixed', verbose=True):
        super().__init__()

        self._schedule_string = schedule_string
        self._print = _print_if_verbose(verbose)

    def on_train_begin(self, logs=None):
        # Make sure that the model is actually using TRADES loss.
        if not isinstance(self.model.loss, Trades):
            raise ValueError(
                '`TradesScheduler` can only be used with `Trades` loss.')

        schedule = np.array(_parse_schedule_string(
            self._schedule_string, self.params['epochs']))

        # We assume the TRADES parameter originally set on the loss is the
        # desired final TRADES parameter, and scale the schedule accordingly.
        self._epoch_to_lam = self.model.loss.lam * schedule

        self.__prev_lam = None
        self.__current_lam = self._epoch_to_lam[0]

    def on_epoch_begin(self, epoch, logs=None):
        if epoch < len(self._epoch_to_lam):
            next_lam = self._epoch_to_lam[epoch]

            self.model.loss.lam = next_lam
        
            if self.__prev_lam is None or self.__prev_lam != self.__current_lam:
                self._print(
                    f'---- setting TRADES parameter={next_lam:.3f} ----')

            self.__prev_lam = self.__current_lam
            self.__current_lam = next_lam


class UpdatePowerIterates(Callback):
    def __init__(
            self, 
            convergence_threshold=1e-4, 
            iteration_batch_size=100, 
            verbose=True):

        super().__init__()

        self._convergence_threshold = convergence_threshold
        self._batch_size = iteration_batch_size

        self._print = _print_if_verbose(verbose)
        self._verbose = verbose

    def on_train_begin(self, logs=None):
        _check_is_gloro_net(self, self.model)

    def on_epoch_begin(self, epoch, logs=None):
        self._print('---- refreshing iterates ----')

        self.model.refresh_iterates(
            convergence_threshold=self._convergence_threshold,
            batch_size=self._batch_size,
            verbose=self._verbose)

    def on_train_end(self, logs=None):
        self._print('---- refreshing iterates ----')

        self.model.refresh_iterates(
            convergence_threshold=self._convergence_threshold,
            batch_size=self._batch_size,
            verbose=self._verbose)

    def on_train_batch_end(self, batch, logs=None):
        self.model.update_iterates()


class LrScheduler(LearningRateScheduler):

    def __init__(self, schedule_string, duration, verbose=True):
        if schedule_string == 'fixed':
            def scheduler(epoch, lr):
                return lr

        elif schedule_string.startswith('decay_to_'):
            end_lr = float(schedule_string.split('decay_to_')[1].split('_')[0])

            if schedule_string.endswith('_after_half'):
                def scheduler(epoch, lr):
                    if epoch < duration // 2:
                        self._initial_lr = lr

                        return lr

                    else:
                        return self._initial_lr * (self._initial_lr / end_lr)**(
                            -(epoch - duration // 2) / (duration // 2))

            else:
                def scheduler(epoch, lr):
                    if epoch == 0:
                        self._initial_lr = lr

                    return self._initial_lr * (self._initial_lr / end_lr)**(
                        -epoch / duration)

        elif (schedule_string.startswith('half_') and 
                schedule_string.endswith('_times')):

            times = int(schedule_string.split('half_')[1].split('_times')[0])
            period = duration // times

            def scheduler(epoch, lr):
                if epoch % period == period - 1:
                    return lr / 2.
                else:
                    return lr

        else:
            raise ValueError(f'unrecognized schedule string: {schedule_string}')

        super().__init__(scheduler, verbose=1 if verbose else 0)
