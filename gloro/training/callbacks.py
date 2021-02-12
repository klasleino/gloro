from tensorflow.keras.callbacks import Callback
from tensorflow.keras.callbacks import LearningRateScheduler

from gloro.models import GloroNet


def _print_if_verbose(verbose):
    if verbose:
        return lambda s: print(s)

    else:
        return lambda s: None


def _check_is_gloro_net(callback, model):
    if not isinstance(model, GloroNet):
        raise ValueError(
            f'`{callback.__class__.__name__}` can only be used with a '
            f'`GloroNet` model')


class EpsilonScheduler(Callback):
    def __init__(self, epsilon, schedule_string, duration, verbose=True):

        super().__init__()

        if schedule_string == 'fixed':
            self._epoch_to_eps = [epsilon]

        elif schedule_string == 'linear_half':
            self._epoch_to_eps = [
                epsilon * i / (duration // 2) for i in range(duration // 2 + 1)
            ]

        elif schedule_string == 'linear':
            self._epoch_to_eps = [
                epsilon * i / (duration - 1) for i in range(duration)
            ]

        elif schedule_string == 'logarithmic':
            initial = 0.01
            final = 1.

            self._epoch_to_eps = [
                epsilon * np.log(
                    (np.exp(final) - np.exp(initial)) / (duration - 1) * i +
                    np.exp(initial))
                for i in range(duration)
            ]

        elif schedule_string == 'logarithmic_half':
            initial = 0.01
            final = 1.

            self._epoch_to_eps = [
                epsilon * np.log(
                    (np.exp(final) - np.exp(initial)) / (duration // 2) * i +
                    np.exp(initial))
                for i in range(duration // 2 + 1)
            ]

        else:
            raise ValueError(f'unrecognized schedule string: {schedule_string}')

        self._print = _print_if_verbose(verbose)

        self.__prev_eps = None
        self.__current_eps = self._epoch_to_eps[0]

    def on_epoch_begin(self, epoch, logs=None):
        _check_is_gloro_net(self, self.model)

        if epoch < len(self._epoch_to_eps):
            next_eps = self._epoch_to_eps[epoch]

            self.model.epsilon = next_eps
        
            if self.__prev_eps is None or self.__prev_eps != self.__current_eps:
                self._print(f'---- setting epsilon={next_eps:.3f} ----')

            self.__prev_eps = self.__current_eps
            self.__current_eps = next_eps


class UpdatePowerIterates(Callback):
    def __init__(
            self, 
            duration, 
            convergence_threshold=1e-4, 
            iteration_batch_size=100, 
            verbose=True):

        super().__init__()

        self._duration = duration
        self._convergence_threshold = convergence_threshold
        self._batch_size = iteration_batch_size

        self._print = _print_if_verbose(verbose)
        self._verbose = verbose

    def on_epoch_begin(self, epoch, logs=None):
        _check_is_gloro_net(self, self.model)

        self._print('\n-- refreshing iterates --')

        self.model.refresh_iterates(
            convergence_threshold=self._convergence_threshold,
            batch_size=self._batch_size,
            verbose=self._verbose)

    def on_epoch_end(self, epoch, logs=None):
        if epoch >= self._duration - 1:
            self._print('\n-- refreshing iterates --')

            self.model.refresh_iterates(
                convergence_threshold=self._convergence_threshold,
                batch_size=self._batch_size,
                verbose=self._verbose)

            self._print(' ')

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
