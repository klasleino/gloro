from gloro.layers import Bias
from gloro.layers import InvertibleDownsampling
from gloro.layers import MinMax
from gloro.layers import Scaling
from gloro.training.losses import Crossentropy
from gloro.training.losses import Trades
from gloro.training.metrics import clean_acc
from gloro.training.metrics import clean_acc_cat
from gloro.training.metrics import clean_acc_sparse
from gloro.training.metrics import vra
from gloro.training.metrics import vra_cat
from gloro.training.metrics import vra_sparse


GLORO_CUSTOM_OBJECTS = {
    'Crossentropy': Crossentropy,
    'Trades': Trades,
    'clean_acc': clean_acc,
    'clean_acc_cat': clean_acc_cat,
    'clean_acc_sparse': clean_acc_sparse,
    'vra': vra,
    'vra_cat': vra_cat,
    'vra_sparse':vra_sparse,

    'MinMax': MinMax,
    'InvertibleDownsampling': InvertibleDownsampling,
    'Scaling': Scaling,
    'Bias': Bias,
}

EPS = 1e-9

MAIN_BRANCH = 'main'
RESIDUAL_BRANCH = 'residual'
SKIP_BRANCH = 'skip'
