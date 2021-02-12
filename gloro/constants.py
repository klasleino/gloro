from gloro import GloroNet
from gloro.training.losses import Crossentropy
from gloro.training.losses import Trades
from gloro.training.metrics import clean_acc
from gloro.training.metrics import vra


GLORO_CUSTOM_OBJECTS = {
    'GloroNet': GloroNet,
    'Crossentropy': Crossentropy,
    'Trades': Trades,
    'clean_acc': clean_acc,
    'vra': vra,
}
