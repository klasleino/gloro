from abc import abstractmethod

from tensorflow.keras.layers import Layer


class GloroLayer(Layer):
    """
    Layers used by GloRo Nets need to provide a way to calculate an upper bound
    on their Lipschitz constant. If you want to provide a custom layer to be
    used in a GloRo Net, it should extend this class.
    """

    @abstractmethod
    def lipschitz(self):
        """
        Returns the Lipschitz constant of this layer.
        """
        raise NotImplementedError
