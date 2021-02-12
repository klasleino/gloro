from gloro import GloroNet
from gloro.training.losses import Crossentropy
from gloro.training.losses import Trades


GLORO_CUSTOM_OBJECTS = {
    'GloroNet': GloroNet,
    'Crossentropy': Crossentropy,
    'Trades': Trades,
}
